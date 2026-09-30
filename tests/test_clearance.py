"""mech.clearance: statuses, containment, overlap volume/extent/location, pair semantics, cache, sweeps."""

from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos, Sphere, Vector
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.Extrema import Extrema_ExtFlag_MIN

from mech import Assembly, Kinematics
from mech.assembly import Study
from mech.clearance import ClearanceChecker, PairResult
from mech.geom import rot_about_line, to_location, translation
from mech.kinematics import Pose
from mech.motion import run_study
from mech.parts import gear_pair

FAR = Pos(0, 0, -500) * Box(4, 4, 4)  # a ground part well away from everything under test


def two_parts(shape_a, shape_b, *, clearance=0.3, joined=False, allowed=False, ignored=False,
              max_depth=0.1) -> Assembly:
    """Parts 'a' and 'b' on slides from a distant base (not joined) or b hinged on a (joined)."""
    asm = Assembly("pair", clearance=clearance)
    asm.part("base", FAR, ground=True)
    asm.part("a", shape_a)
    asm.part("b", shape_b)
    asm.prismatic("j_a", "base", "a", origin=(0, 0, 0), axis=(1, 0, 0))
    if joined:
        asm.revolute("j_b", "a", "b", origin=(0, 0, 0), axis=(0, 0, 1))
    else:
        asm.prismatic("j_b", "base", "b", origin=(0, 0, 0), axis=(1, 0, 0))
    if allowed:
        asm.allow_contact("a", "b", max_depth=max_depth)
    if ignored:
        asm.ignore("a", "b")
    return asm


def pair_result(results: list[PairResult], a: str = "a", b: str = "b") -> PairResult | None:
    found = [r for r in results if {r.a, r.b} == {a, b}]
    assert len(found) <= 1
    return found[0] if found else None


CUBE10 = Pos(5, 5, 5) * Box(10, 10, 10)  # [0, 10]³


def cube10_at_gap(g: float):
    """A second 10 mm cube, g mm beyond CUBE10 along +x."""
    return Pos(15 + g, 5, 5) * Box(10, 10, 10)


# ------------------------------------------------------------------------------ containment & overlap


@pytest.mark.parametrize("outer_first", [True, False])
def test_containment_is_interference(outer_first):
    """A 4 mm cube buried in a 20 mm cube: OCC's distance sees an 8 mm gap; containment makes it 64 mm³."""
    big, small = Box(20, 20, 20), Pos(1, 2, 3) * Box(4, 4, 4)
    asm = two_parts(big, small) if outer_first else two_parts(small, big)
    r = pair_result(ClearanceChecker(asm).check_pose({}))
    assert r.status == "interference" and r.distance == 0.0
    assert r.volume == pytest.approx(64.0, rel=0.01)
    np.testing.assert_allclose(r.extent, [4, 4, 4], atol=1e-6)
    np.testing.assert_allclose(r.location, [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(r.pa, [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(r.pb, r.pa)


def test_overlapping_boxes_volume_extent_location():
    b = Pos(13, 7, 2) * Box(10, 10, 10)  # [8, 18] × [2, 12] × [−3, 7]
    checker = ClearanceChecker(two_parts(CUBE10, b))
    r = pair_result(checker.check_pose({}))
    assert r.status == "interference" and r.distance == 0.0 and not r.joined
    assert r.volume == pytest.approx(2 * 8 * 7, rel=1e-9)  # [8, 10] × [2, 10] × [0, 7]
    np.testing.assert_allclose(r.extent, [2, 8, 7], atol=1e-6)
    np.testing.assert_allclose(r.location, [9, 6, 3.5], atol=1e-6)

    # move a, and b 1 mm further into a (in a's frame): location stays in a's HOME frame, pa/pb are current world
    Ta = translation((30, -20, 5)) @ rot_about_line((0, 0, 0), (1, 1, 1), 40)
    moved = pair_result(checker.check_pose({"a": Ta, "b": Ta @ translation((-1, 0, 0))}))
    assert moved.volume == pytest.approx(3 * 8 * 7, rel=1e-9)  # [7, 10] × [2, 10] × [0, 7]
    np.testing.assert_allclose(moved.extent, [3, 8, 7], atol=1e-6)
    np.testing.assert_allclose(moved.location, [8.5, 6, 3.5], atol=1e-6)
    np.testing.assert_allclose(moved.pa, Ta[:3, :3] @ [8.5, 6, 3.5] + Ta[:3, 3], atol=1e-6)


# ------------------------------------------------------------------------------ statuses


@pytest.mark.parametrize("gap, status", [(0.2, "tight"), (0.29, "tight"), (0.3, "ok"), (0.5, "ok"), (2.0, "ok")])
def test_gap_thresholds(gap, status):
    r = pair_result(ClearanceChecker(two_parts(CUBE10, cube10_at_gap(gap))).check_pose({}))
    # the only non-joined pair is always measured exactly (it is the minimum clearance)
    assert r.status == status and r.volume == 0.0 and r.extent is None
    assert r.distance == pytest.approx(gap, abs=1e-9)
    assert r.pa[0] == pytest.approx(10.0, abs=1e-9) and r.pb[0] == pytest.approx(10.0 + gap, abs=1e-9)
    assert r.location[0] == pytest.approx(10.0 + gap / 2, abs=1e-9)


def test_touching_is_tight_unless_joined_or_allowed():
    touching = cube10_at_gap(0.0)
    r = pair_result(ClearanceChecker(two_parts(CUBE10, touching)).check_pose({}))
    assert (r.status, r.distance, r.volume) == ("tight", 0.0, 0.0)
    for kw in ({"joined": True}, {"allowed": True}):
        r = pair_result(ClearanceChecker(two_parts(CUBE10, touching, **kw)).check_pose({}))
        assert (r.status, r.distance, r.volume) == ("contact", 0.0, 0.0)
    # joined pairs are never tight: a 0.2 mm gap is fine and needs no exact measurement
    assert pair_result(ClearanceChecker(two_parts(CUBE10, cube10_at_gap(0.2), joined=True)).check_pose({})) is None


def test_overlap_tolerances_and_pair_semantics():
    shallow = cube10_at_gap(-0.004)  # 0.4 mm³ overlap, mean depth ≈ 0.004 mm: within both tolerances
    r = pair_result(ClearanceChecker(two_parts(CUBE10, shallow)).check_pose({}))
    assert r.status == "tight" and r.volume == pytest.approx(0.4, rel=1e-6)
    assert pair_result(ClearanceChecker(two_parts(CUBE10, shallow, joined=True)).check_pose({})).status == "contact"

    deep_thin = Pos(10.2, 5, 5) * Box(1, 1, 1)  # 1×1 rod 0.3 mm into the cube: 0.3 mm³ < tol ...
    r = pair_result(ClearanceChecker(two_parts(CUBE10, deep_thin)).check_pose({}))
    assert r.volume == pytest.approx(0.3, rel=1e-6)
    assert 2 * r.volume / (2 * 1 + 4 * 0.3) > 0.02 and r.status == "interference"  # ... but 0.19 mm deep

    big = cube10_at_gap(-1.0)  # 100 mm³, mean depth 2V/A = 200 / 240 = 0.833 mm
    for kw, status in (({}, "interference"), ({"joined": True}, "interference"),
                       ({"allowed": True, "max_depth": 1.0}, "contact")):
        r = pair_result(ClearanceChecker(two_parts(CUBE10, big, **kw)).check_pose({}))
        assert r.status == status and r.volume == pytest.approx(100.0, rel=1e-9)
        assert r.depth == pytest.approx(200 / 240, rel=1e-9)
        assert (r.joined, r.allowed) == (kw.get("joined", False), kw.get("allowed", False))
    assert pair_result(ClearanceChecker(two_parts(CUBE10, big, ignored=True)).check_pose({})) is None


def gear_model() -> tuple[Assembly, float]:
    gp = gear_pair(1, 15, 30, 6)
    asm = Assembly("gears")
    asm.part("frame", Pos(11, 0, -6) * Box(60, 20, 4), ground=True)
    asm.part("g1", gp.g1)
    asm.part("g2", gp.g2)
    asm.revolute("j1", "frame", "g1", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j2", "frame", "g2", origin=(gp.center_distance, 0, 0), axis=(0, 0, 1))
    asm.gear("j1", "j2", gp.ratio)
    return asm, gp.center_distance


def test_meshing_pair_is_interference_only():
    asm, C = gear_model()
    checker = ClearanceChecker(asm)
    home = pair_result(checker.check_pose({}), "g1", "g2")
    assert home.meshing and home.distance is None and home.status in ("ok", "contact") and home.volume == 0.0
    # in mesh: the gear ratio keeps the teeth apart through a sweep
    kin = Kinematics(asm)
    sweep = checker.sweep(run_study(asm, kin, Study("mesh", {"j1": (0, 30)}, frames=7)))
    assert all(r.status in ("ok", "contact") for frame in sweep.per_frame for r in frame)
    assert sweep.min_clearance is None  # every pair is joined or meshing
    # g2 turned 2° off the ratio: the teeth collide; volume equals the boolean common
    T2 = rot_about_line((C, 0, 0), (0, 0, 1), 2.0)
    r = pair_result(checker.check_pose({"g2": T2}), "g1", "g2")
    expected = (asm.parts["g1"].shape & asm.parts["g2"].shape.moved(to_location(T2))).volume
    assert r.status == "interference" and r.distance is None
    assert expected > 1.0 and r.volume == pytest.approx(expected, rel=1e-6)


def test_rigid_group_pairs_are_checked_at_home_only():
    asm = Assembly("rigid")
    asm.part("base", Box(20, 20, 4), ground=True)
    asm.part("bracket", Pos(5, 0, 3) * Box(6, 6, 4))  # sinks 1 mm into the base: 6·6·1 = 36 mm³
    asm.part("arm", Pos(20, 0, 10) * Box(20, 4, 4))
    asm.fix("bracket", "base")
    asm.revolute("j_arm", "base", "arm", origin=(10, 0, 10), axis=(0, 0, 1))
    checker = ClearanceChecker(asm)
    r = pair_result(checker.check_pose({}), "base", "bracket")
    assert r.status == "interference" and r.joined and r.volume == pytest.approx(36.0, rel=1e-9)
    assert pair_result(checker.check_pose({}, include_rigid=False), "base", "bracket") is None
    sweep = checker.sweep(run_study(asm, Kinematics(asm), Study("s", {"j_arm": (0, 90)}, frames=4)))
    assert all({r.a, r.b} != {"base", "bracket"} for frame in sweep.per_frame for r in frame)


def test_open_frames_are_not_checked():
    asm = two_parts(CUBE10, cube10_at_gap(5.0))
    overlapping = {"base": np.eye(4), "a": np.eye(4), "b": translation((-8, 0, 0))}
    result = SimpleNamespace(poses=[Pose({}, overlapping, 1.0, False), Pose({}, overlapping, 0.0, True)])
    sweep = ClearanceChecker(asm).sweep(result)
    assert sweep.per_frame[0] == [] and sweep.per_frame[1][0].status == "interference"
    # signed: −(mean depth 2V/A) of the 3×10×10 overlap = −600/320 mm
    assert sweep.stats["frames_skipped"] == 1 and sweep.min_clearance == ("a", "b", pytest.approx(-1.875), 1)


# ------------------------------------------------------------------------------ cache & sweeps


def test_cache_maps_rigid_repeats_through_the_current_pose():
    """Two spheres turning together: one exact measurement, closest points follow every frame."""
    asm = Assembly("spheres", clearance=3.0)
    asm.part("base", FAR, ground=True)
    asm.part("a", Pos(20, 0, 0) * Sphere(5))
    asm.part("b", Pos(20, 12, 0) * Sphere(5))  # 2 mm gap, closest points on the line of centers
    asm.revolute("j_a", "base", "a", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j_b", "base", "b", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.couple("j_a", "j_b", ratio=1.0)
    res = run_study(asm, Kinematics(asm), Study("turn", {"j_a": (0, 90)}, frames=5))
    sweep = ClearanceChecker(asm).sweep(res)
    assert sweep.stats["cache_hits"] == 4 and sweep.stats["exact"] == 1
    for k, frame in enumerate(sweep.per_frame):
        (r,) = frame
        R = rot_about_line((0, 0, 0), (0, 0, 1), 22.5 * k)[:3, :3]
        assert r.status == "tight" and r.distance == pytest.approx(2.0, abs=1e-7)
        np.testing.assert_allclose(r.pa, R @ [20, 5, 0], atol=1e-6)
        np.testing.assert_allclose(r.pb, R @ [20, 7, 0], atol=1e-6)
        np.testing.assert_allclose(r.location, [20, 6, 0], atol=1e-6)  # home world, independent of k
    assert sweep.min_clearance[:2] == ("a", "b") and sweep.min_clearance[2] == pytest.approx(2.0, abs=1e-7)
    assert sweep.min_clearance[3] == 0


def _clip_above(poly: np.ndarray, y0: float) -> np.ndarray:
    """Sutherland–Hodgman: the part of a convex polygon with y ≥ y0."""
    out = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        if p[1] >= y0:
            out.append(p)
        if (p[1] >= y0) != (q[1] >= y0):
            out.append(p + (y0 - p[1]) / (q[1] - p[1]) * (q - p))
    return np.array(out)


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def arm_and_wall(clearance: float) -> Assembly:
    """A 50×4×4 arm turning about Z toward a wall at y = 40 (held on a slide, so not joined)."""
    asm = Assembly("arm_wall", clearance=clearance)
    asm.part("base", FAR, ground=True)
    asm.part("wall", Pos(0, 50, 2.5) * Box(120, 20, 15))  # y ∈ [40, 60], z ∈ [−5, 10]
    asm.part("arm", Pos(25, 0, 2) * Box(50, 4, 4))  # x ∈ [0, 50], y ∈ [−2, 2], z ∈ [0, 4]
    asm.prismatic("j_wall", "base", "wall", origin=(0, 40, 0), axis=(1, 0, 0))
    asm.revolute("j_arm", "base", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
    return asm


def arm_gap(theta_deg: float) -> float:
    t = math.radians(theta_deg)
    return 40.0 - (50.0 * math.sin(t) + 2.0 * math.cos(t))  # the corner (50, 2) is the arm's highest point


def test_sweep_over_revolute_study():
    asm = arm_and_wall(clearance=0.5)
    res = run_study(asm, Kinematics(asm), Study("swing", {"j_arm": (0, 60)}, frames=13))  # 5° steps
    sweep = ClearanceChecker(asm).sweep(res)
    assert [len(f) for f in sweep.per_frame] == [0] * 10 + [1, 1, 1]
    tight = sweep.per_frame[10][0]
    assert (tight.a, tight.b, tight.status) == ("wall", "arm", "tight")
    assert tight.distance == pytest.approx(arm_gap(50.0), abs=1e-7) and 0 < arm_gap(50.0) < 0.5
    rect = np.array([[0.0, -2.0], [50.0, -2.0], [50.0, 2.0], [0.0, 2.0]])
    depths = {}
    for k in (11, 12):
        R = rot_about_line((0, 0, 0), (0, 0, 1), 5.0 * k)[:2, :2]
        overlap = _clip_above(rect @ R.T, 40.0)
        r = sweep.per_frame[k][0]
        assert r.status == "interference" and r.distance == 0.0
        assert r.volume == pytest.approx(4.0 * _area(overlap), rel=1e-7)
        lo, hi = overlap.min(axis=0), overlap.max(axis=0)
        np.testing.assert_allclose(r.extent, [*(hi - lo), 4.0], atol=1e-6)
        np.testing.assert_allclose(r.location, [*((lo + hi) / 2), 2.0], atol=1e-6)  # wall = a: home = world
        # the arm's home frame: the same point turned back by the arm's rotation
        np.testing.assert_allclose(r.location_b, [*(R.T @ ((lo + hi) / 2)), 2.0], atol=1e-6)
        perimeter = float(np.sum(np.linalg.norm(np.roll(overlap, -1, axis=0) - overlap, axis=1)))
        depths[k] = 2 * 4.0 * _area(overlap) / (2 * _area(overlap) + 4.0 * perimeter)  # 2V/A of the prism
        assert r.depth == pytest.approx(depths[k], rel=1e-6)
    assert sweep.worst[frozenset(("wall", "arm"))][0] == 12  # the larger overlap
    # signed minimum clearance: the deepest overlap, negative
    assert sweep.min_clearance == ("wall", "arm", pytest.approx(-depths[12], rel=1e-6), 12)
    assert sweep.stats["pairs_checked"] > 0 and sweep.stats["seconds"] > 0 and sweep.stats["frames"] == 13


def test_min_clearance_is_exact_beyond_the_tight_threshold():
    """Nothing is tight at clearance 0.1, yet the minimum clearance is still the exact 50° gap."""
    asm = arm_and_wall(clearance=0.1)
    res = run_study(asm, Kinematics(asm), Study("swing", {"j_arm": (0, 50)}, frames=11))
    sweep = ClearanceChecker(asm).sweep(res)
    assert all(frame == [] for frame in sweep.per_frame) and sweep.worst == {}
    a, b, gap, frame = sweep.min_clearance
    assert (a, b, frame) == ("wall", "arm", 10) and gap == pytest.approx(arm_gap(50.0), abs=1e-7)
    # a single pose too: the returned list holds the exact closest pair
    home = ClearanceChecker(asm).check_pose({})
    assert pair_result(home, "wall", "arm").distance == pytest.approx(arm_gap(0.0), abs=1e-7)


def _ring(r_in: float) -> object:
    body = Pos(0, 0, 5) * (Cylinder(r_in + 2, 10) - Cylinder(r_in, 10))
    return body + Pos(r_in + 4, 0, 5) * Box(4, 3, 10)  # a tab breaks the rotational symmetry


def test_nested_parts_min_clearance_matches_brute_force():
    """Nested rings (every AABB overlaps) turning at different rates: the prefiltered search
    finds the same minimum as exact distances over every pair and frame."""
    asm = Assembly("rings", clearance=0.3)
    asm.part("base", FAR, ground=True)
    for i in range(3):
        asm.part(f"ring{i}", _ring(5.0 + 8.0 * i))
        asm.revolute(f"j{i}", "base", f"ring{i}", origin=(0, 0, 0), axis=(0, 0, 1))
        if i:
            asm.couple("j0", f"j{i}", ratio=1.0 + 0.37 * i)
    res = run_study(asm, Kinematics(asm), Study("turn", {"j0": (0, 120)}, frames=4))
    sweep = ClearanceChecker(asm).sweep(res)
    brute = []
    for k, pose in enumerate(res.poses):
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = f"ring{i}", f"ring{j}"
                sa = asm.parts[a].shape.moved(to_location(pose.transforms[a]))
                sb = asm.parts[b].shape.moved(to_location(pose.transforms[b]))
                d = BRepExtrema_DistShapeShape(sa.wrapped, sb.wrapped, Extrema_ExtFlag_MIN).Value()
                brute.append((d, k, a, b))
    d, k, a, b = min(brute)
    assert d > 1.0 and all(f == [] for f in sweep.per_frame)
    assert sweep.min_clearance[:2] == (a, b) and sweep.min_clearance[3] == k
    assert sweep.min_clearance[2] == pytest.approx(d, abs=1e-9)


# ------------------------------------------------------------------------------ review regressions


def test_near_miss_against_another_ground_part_is_tight():
    """A blade hinged to the ground base swings 0.05 mm past a separate ground post (review
    p2_ground_joined): the post is not what the hinge connects, so the gap is checked."""
    asm = Assembly("groundjoined", clearance=0.3)
    asm.part("base", Pos(0, 0, -10) * Box(120, 120, 4), ground=True)
    asm.part("post", Pos(40, 2.05, -0.95) * Box(2, 2, 13.9), ground=True)
    asm.part("blade", Pos(25, 0, 2.5) * Box(50, 2, 5))
    asm.revolute("j", "base", "blade", origin=(0, 0, 0), axis=(0, 0, 1), limits=(-30, 0))
    sweep = ClearanceChecker(asm).sweep(run_study(asm, Kinematics(asm), Study("s", {"j": (-30, 0)}, frames=31)))
    k, r = sweep.worst[frozenset(("post", "blade"))]
    assert (r.status, k) == ("tight", 30) and r.distance == pytest.approx(0.05, abs=1e-9)
    assert sweep.min_clearance[:2] == ("post", "blade") and sweep.min_clearance[2] == pytest.approx(0.05, abs=1e-9)
    # the pair the hinge names stays exempt unless the model asks for it
    assert frozenset(("base", "blade")) not in sweep.worst
    asm.check_clearance("base", "blade")
    (pair,) = [p for p in ClearanceChecker(asm)._pairs if {p.a, p.b} == {"base", "blade"}]
    assert not pair.joined and not pair.excused


def test_list_of_shapes_part_is_fused_for_interference():
    """Overlapping list members (review p4_compound / min_overlap_list): volumes count the union,
    and a cube buried in the second member only is still found."""
    jaw = [Box(10, 10, 10), Pos(8, 0, 0) * Box(10, 10, 10)]  # x ∈ [−5, 13], members overlap on [3, 5]
    r = pair_result(ClearanceChecker(two_parts(jaw, Pos(16, 0, 0) * Box(10, 10, 10))).check_pose({}))
    assert r.status == "interference" and r.volume == pytest.approx(2 * 10 * 10, rel=1e-9)  # x ∈ [11, 13]
    r = pair_result(ClearanceChecker(two_parts(jaw, Pos(10, 0, 0) * Box(2, 2, 2))).check_pose({}))
    assert r.status == "interference" and r.volume == pytest.approx(8.0, rel=1e-9)


def test_containment_in_a_multi_solid_library_part():
    """OCC's classifier on the 608's three-solid compound calls its outer ring 'outside'; the
    checker classifies per solid, so a cube buried in the ring interferes (review p5d)."""
    from mech.parts import bearing

    ring = bearing("608").shape
    assert not ring.is_inside(Vector(10.0, 0, 0))  # the compound-level classifier's blind spot
    r = pair_result(ClearanceChecker(two_parts(ring, Pos(10, 0, 0) * Box(0.5, 0.5, 0.5))).check_pose({}))
    assert r.status == "interference" and r.volume == pytest.approx(0.125, rel=1e-6)


def test_allowed_contact_is_bounded_by_max_depth():
    """An allow_contact pair may overlap only as deep as its max_depth (review p13 / gripper)."""
    fit = cube10_at_gap(-0.05)  # 5 mm³, mean depth 10/202 = 0.0495 mm: a snug fit
    deep = cube10_at_gap(-2.0)  # 200 mm³, mean depth 400/280 = 1.43 mm: a jaw driven through the other
    r = pair_result(ClearanceChecker(two_parts(CUBE10, fit, allowed=True)).check_pose({}))
    assert r.status == "contact" and r.depth == pytest.approx(10 / 202, rel=1e-6)
    r = pair_result(ClearanceChecker(two_parts(CUBE10, deep, allowed=True)).check_pose({}))
    assert r.status == "interference" and r.depth == pytest.approx(400 / 280, rel=1e-6)
    assert r.volume == pytest.approx(200.0, rel=1e-9) and r.allowed
    # max_depth=None: any overlap is contact, and no boolean is run for the pair
    checker = ClearanceChecker(two_parts(CUBE10, deep, allowed=True, max_depth=None))
    stats = Counter()
    results, _, _ = checker._check({}, checker._pairs, math.inf, stats)
    r = pair_result(results)
    assert r.status == "contact" and r.depth is None and stats["booleans"] == 0
    # a bounded pair in contact is settled by the boolean alone (no exact distance query)
    checker = ClearanceChecker(two_parts(CUBE10, fit, allowed=True))
    stats = Counter()
    checker._check({}, checker._pairs, math.inf, stats)
    assert stats["booleans"] == 1 and stats["exact_after_boolean"] == 0


def test_min_clearance_is_negative_when_any_pair_interferes():
    """A joined pair driven into its parent: the study's clearance can't read as the big gap of the
    unjoined pairs (review dogfood_pantilt zt=65)."""
    asm = Assembly("sink", clearance=0.3)
    asm.part("base", FAR, ground=True)
    asm.part("plate", Pos(0, 0, -2) * Box(60, 60, 4))  # z ∈ [−4, 0]
    asm.part("arm", Pos(25, 0, 2) * Box(50, 4, 4))  # z ∈ [0, 4], hinged on the plate about +Y
    asm.part("far", Pos(0, 80, 2) * Box(4, 4, 4))
    asm.prismatic("j_plate", "base", "plate", origin=(0, 0, 0), axis=(1, 0, 0))
    asm.prismatic("j_far", "base", "far", origin=(0, 80, 0), axis=(1, 0, 0))
    asm.revolute("j_arm", "plate", "arm", origin=(0, 0, 2), axis=(0, -1, 0))
    res = run_study(asm, Kinematics(asm), Study("dip", {"j_arm": (0, -10)}, frames=3))  # tip sinks into the plate
    sweep = ClearanceChecker(asm).sweep(res)
    k, r = sweep.worst[frozenset(("plate", "arm"))]
    assert r.status == "interference" and r.joined and k == 2
    a, b, value, frame = sweep.min_clearance
    assert (a, b, frame) == ("plate", "arm", 2) and value == pytest.approx(-r.depth) and value < 0


def test_collision_between_frames_is_found():
    """A thin blade passes a post between two sampled frames (review p1_tunnel): both frames are
    clear, the sub-frame check finds the hit and reports it at the nearest frame."""
    asm = Assembly("tunnel", clearance=0.3)
    asm.part("base", FAR, ground=True)
    phi = math.radians(3.75)  # halfway between the 0° and 7.5° frames
    asm.part("post", Pos(40 * math.cos(phi), 40 * math.sin(phi), 2.5) * Box(2, 2, 10), ground=True)
    asm.part("blade", Pos(25, 0, 2.5) * Box(50, 2, 5))
    asm.revolute("j", "base", "blade", origin=(0, 0, 0), axis=(0, 0, 1))
    res = run_study(asm, Kinematics(asm), Study("swing", {"j": (0, 90)}, frames=13))
    checker = ClearanceChecker(asm)
    for k in (0, 1):  # neither sampled frame touches the post
        assert pair_result(checker.check_pose(res.poses[k].transforms), "post", "blade").status == "ok"
    sweep = checker.sweep(res)
    k, r = sweep.worst[frozenset(("post", "blade"))]
    assert r.status == "interference" and r.at == pytest.approx(0.5) and k in (0, 1)
    assert sweep.min_clearance[:2] == ("post", "blade") and sweep.min_clearance[2] < 0
    assert sweep.stats["sub_poses"] >= 1


def test_spinning_disc_over_a_plate_needs_no_sub_frames():
    """Rotation never moves points along its own axis: a disc spinning fast 0.5 mm over a plate is
    settled without intermediate poses, however coarse the frames."""
    asm = Assembly("spin", clearance=0.3)
    asm.part("base", FAR, ground=True)
    asm.part("plate", Pos(0, 0, -2) * (Box(80, 80, 4) - Cylinder(8, 10)), ground=True)  # clear of the axis
    asm.part("disc", Pos(10, 0, 2) * Box(40, 6, 3))  # an arm spinning 0.5 mm above the plate
    asm.revolute("j", "base", "disc", origin=(0, 0, 0), axis=(0, 0, 1))
    assert not asm.is_joined("plate", "disc")
    sweep = ClearanceChecker(asm).sweep(run_study(asm, Kinematics(asm), Study("spin", {"j": (0, 360)}, frames=6)))
    assert sweep.stats.get("sub_poses", 0) == 0 and sweep.worst == {}
    assert sweep.min_clearance[2] == pytest.approx(0.5, abs=1e-9)
