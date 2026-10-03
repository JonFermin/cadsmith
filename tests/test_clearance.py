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
    # check_clearance holds it to clearance — outside the hinge's carried region (a joined twin
    # still checks the whole parts for overlap)
    asm.check_clearance("base", "blade")
    twin, held = sorted((p for p in ClearanceChecker(asm)._pairs if {p.a, p.b} == {"base", "blade"}),
                        key=lambda p: not p.joined)
    assert twin.joined and not held.joined and not held.excused and held.ka and held.kb


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


# ------------------------------------------------------------------------------ boolean validation (A1)


def _desktop_arm_elbow() -> Assembly:
    """The elbow motor and ø5-bore GT2 pinion of examples/showcase/desktop_arm.py, drawn where the
    showcase draws them (the boolean artefact below depends on these exact coordinates)."""
    from build123d import Rot

    from mech.parts import gt2_pulley, nema17

    me = (-55.53822212612591, 68.3978982977259)  # the elbow motor axis (x, z)
    asm = Assembly("elbow", clearance=0.3)
    asm.part("motor", Pos(me[0], -23.0, me[1]) * Rot(90, 0, 0) * nema17(40).shape, ground=True)
    asm.part("pinion", Pos(me[0], -40.0, me[1]) * Rot(-90, 0, 0) * gt2_pulley(20, 6, 5).shape)
    asm.revolute("j", "motor", "pinion", origin=(me[0], -40.0, me[1]), axis=(0, -1, 0))
    return asm


# pinion pose in the motor's frame at frame 78 of desktop_arm's pick_place study, where OCC's
# common of the coincident ø5 shaft and ø5 bore returns the whole shaft section (118 mm³)
_A1_POSE = np.array([[0.727373641573049, -1.152760928809413e-17, -0.6862416378687333, 31.796302502857632],
                     [9.265774377980021e-18, 1.0, 1.2271492831577166e-17, -2.1788514756475548e-16],
                     [0.6862416378687333, 0.0, 0.727373641573049, 56.75971045311613],
                     [0.0, 0.0, 0.0, 1.0]])


def test_coincident_fit_boolean_artefact_is_not_interference():
    """A nominal ø5 bore on the ø5 motor shaft (review A1): the boolean's phantom shaft-sized cell
    holds no point inside the pinion, so it is dropped and the pair only touches."""
    from mech.clearance import _common_cells

    checker = ClearanceChecker(_desktop_arm_elbow())
    (pair,) = checker._pairs
    assert pair.joined and (pair.a, pair.b) == ("motor", "pinion")
    moved = checker._bodies["pinion"].shape.Moved(to_location(_A1_POSE).wrapped)
    raw = sum(c.volume for c in _common_cells(checker._bodies["motor"].shape, moved))
    stats = Counter()
    assert checker._common(pair, _A1_POSE, None, None, stats) is None  # the whole-part boolean, validated
    if raw > 1.0:  # OCC still makes the artefact at this pose: validation removed it
        assert stats["spurious_cells"] >= 1
    r = pair_result(checker.check_pose({"pinion": _A1_POSE}), "motor", "pinion")
    assert r is not None and r.status == "contact" and r.volume == 0.0


def test_spurious_cell_is_dropped_and_genuine_cell_kept(monkeypatch):
    """Validation on a forced artefact: a cell lying in one part only is dropped (and the swapped
    boolean consulted); the ring of a genuine 0.2 mm press fit stands."""
    import mech.clearance as C

    shaft = Pos(0, 0, -10) * Cylinder(2.5, 40)
    asm = two_parts(shaft, Pos(0, 0, 3) * (Cylinder(10, 6) - Cylinder(2.5, 6)), joined=True)  # nominal ø5 bore
    checker = ClearanceChecker(asm)
    pair = next(p for p in checker._pairs if {p.a, p.b} == {"a", "b"})
    phantom = Pos(0, 0, 3) * Cylinder(2.5, 6)  # the shaft inside the bore: in a, not in b
    real = C._common_cells
    phantom_cells = real(phantom.wrapped, phantom.wrapped)
    calls = []

    def fake(sa, sb):
        calls.append((sa, sb))
        return phantom_cells if len(calls) == 1 else []

    monkeypatch.setattr(C, "_common_cells", fake)
    stats = Counter()
    assert checker._common(pair, np.eye(4), None, None, stats) is None
    assert stats["spurious_cells"] == 1 and len(calls) == 2  # dropped, then the swapped boolean
    monkeypatch.setattr(C, "_common_cells", real)
    tight = Pos(0, 0, 3) * (Cylinder(10, 6) - Cylinder(2.3, 6))
    r = pair_result(ClearanceChecker(two_parts(shaft, tight, joined=True)).check_pose({}))
    assert r.status == "interference" and r.volume == pytest.approx(math.pi * (2.5**2 - 2.3**2) * 6, rel=1e-6)


def test_press_fit_interferes_at_every_angle_and_is_measured_once():
    """A 0.2 mm press fit of a round pin turning in a lever: interference in every frame, and the
    pin's rotational symmetry makes every frame one measurement (B4)."""
    asm = Assembly("press", clearance=0.3)
    lever = Pos(10, 0, 3) * Box(40, 12, 6) - Pos(0, 0, 3) * Cylinder(2.3, 6)  # ø4.6 bore at the origin
    asm.part("lever", lever, ground=True)
    asm.part("pin", Pos(0, 0, 3) * Cylinder(2.5, 20))
    asm.revolute("j", "lever", "pin", origin=(0, 0, 0), axis=(0, 0, 1))
    sweep = ClearanceChecker(asm).sweep(run_study(asm, Kinematics(asm), Study("turn", {"j": (0, 350)}, frames=36)))
    expected = math.pi * (2.5**2 - 2.3**2) * 6
    assert all(len(f) == 1 and f[0].status == "interference" for f in sweep.per_frame)
    assert all(f[0].volume == pytest.approx(expected, rel=1e-6) for f in sweep.per_frame)
    assert sweep.stats["booleans"] == 1 and sweep.stats["cache_hits"] == 35


def test_nominal_fit_on_a_motor_shaft_never_interferes_while_turning():
    """The A1 pair turned through a full revolution in 5° steps: never an interference, and the
    round shaft solid makes the shaft/bore boolean a single measurement."""
    asm = _desktop_arm_elbow()
    sweep = ClearanceChecker(asm).sweep(run_study(asm, Kinematics(asm), Study("turn", {"j": (0, 355)}, frames=72)))
    assert all(r.status == "contact" for _, r in sweep.worst.values())
    assert sweep.stats["booleans"] <= 2


# ------------------------------------------------------------------------------ narrowphase vs brute force (B2)


@pytest.mark.parametrize("which", ["gear_pulley", "motor_pulley", "nested"])
def test_narrowphase_matches_brute_force_on_random_poses(which):
    """For random relative poses the pipeline (box bounds, proximity, exact distance on the near
    faces, containment) never claims more gap than OCC's whole-part exact distance (no false
    negative) and, where it measures, agrees with it."""
    from mech.clearance import _common_cells
    from mech.massprops import solid_body
    from mech.parts import gt2_pulley, nema17, spur_gear

    shapes = {"gear_pulley": (spur_gear(1, 15, 6).shape, gt2_pulley(20, 6, 5).shape),
              "motor_pulley": (nema17().shape, gt2_pulley(20, 6, 5).shape),
              "nested": (Box(30, 30, 30) - Box(26, 26, 26), Pos(3, 0, 0) * Box(4, 4, 4))}[which]
    checker = ClearanceChecker(two_parts(*shapes))
    pair = next(p for p in checker._pairs if {p.a, p.b} == {"a", "b"})
    sa, sb = solid_body(shapes[0]), solid_body(shapes[1])
    rng = np.random.default_rng(7)
    for _ in range(5 if which == "gear_pulley" else 12):  # (the reference distance is slow on gear teeth)
        M = rot_about_line((0, 0, 0), rng.normal(size=3), rng.uniform(0, 360))
        M[:3, 3] = rng.normal(size=3) * (4.0 if which == "nested" else 9.0)
        tau = float(rng.choice([1e-6, 0.3, 2.0]))
        m, lb = checker._lookup(pair, M, tau, Counter())
        moved = sb.moved(to_location(M))
        overlap = sum(c.volume for c in _common_cells(sa.wrapped, moved.wrapped))  # the whole parts' boolean
        exact = BRepExtrema_DistShapeShape(sa.wrapped, moved.wrapped, Extrema_ExtFlag_MIN)
        d = 0.0 if overlap > 1e-9 else exact.Value()
        if m is None:
            assert overlap <= 1e-9 and d >= tau - 1e-9 and lb <= d + 1e-9  # a valid lower bound
        else:
            assert m.distance == pytest.approx(d, abs=1e-7)
            assert m.volume == pytest.approx(overlap, rel=1e-4, abs=1e-9)  # (a boolean on the touching solids)


# ------------------------------------------------------------------------------ sweep stats, guard, exemption


def test_sweep_stats_and_progress():
    """B5: the stats contract keys, the slowest pairs, and one progress call per frame."""
    asm = arm_and_wall(clearance=0.5)
    calls = []
    res = run_study(asm, Kinematics(asm), Study("swing", {"j_arm": (0, 60)}, frames=7))
    sweep = ClearanceChecker(asm).sweep(res, progress=lambda *a: calls.append(a))
    assert calls == [("clearance", k, 7) for k in range(1, 8)]
    st = sweep.stats
    for key in ("seconds", "pairs", "exact", "booleans", "subframe_poses", "proximity", "slowest"):
        assert key in st
    assert st["pairs"] == 3 and st["seconds"] > 0 and st["exact"] >= 1
    assert 1 <= len(st["slowest"]) <= 5 and all(len(row) == 3 and row[2] >= 0 for row in st["slowest"])
    assert ("wall", "arm") in {(a, b) for a, b, _ in st["slowest"]}


@pytest.mark.parametrize("slide", [False, True])
def test_enclosed_part_far_from_the_walls_settles_without_sub_frames(slide):
    """B1: an arm tumbling inside a hollow housing (the boxes overlap, the walls are 20 mm away):
    the frames' gaps are proven large enough for the motion, so nothing is bisected — whether the
    housing shares the arm's hinge line (a gap held at every pose) or sits on a slide."""
    asm = Assembly("cage", clearance=0.3)
    asm.part("base", FAR, ground=True)
    asm.part("housing", Box(100, 100, 30) - Box(90, 90, 40), ground=not slide)
    asm.part("arm", Pos(12, 0, 0) * Box(24, 4, 4))
    asm.revolute("j", "base", "arm", origin=(0, 0, 0), axis=(1, 1, 1))  # no axial separation to lean on
    if slide:
        asm.prismatic("j_slide", "base", "housing", origin=(0, 0, 0), axis=(1, 0, 0))
    checker = ClearanceChecker(asm)
    assert (checker._hinge_line("housing", "arm") is None) == slide
    sweep = checker.sweep(run_study(asm, Kinematics(asm), Study("spin", {"j": (0, 80)}, frames=3)))
    assert sweep.worst == {} and sweep.stats["subframe_poses"] == 0


def _co_hinged(lug: float | None) -> Assembly:
    """Two arms hinged to the ground at one pivot, in layers 0.8 mm apart; the upper arm's hub
    drops through a ø6 hole of the lower one (1 mm radial gap). With ``lug`` (deg) the upper arm
    also has a block hanging into the lower arm's layer, 40 mm out at that angle."""
    asm = Assembly("co_hinged", clearance=0.5)
    asm.part("base", FAR, ground=True)
    asm.part("lower", Pos(25, 0, 1.5) * Box(50, 6, 3) - Pos(0, 0, 1.5) * Cylinder(3, 3))  # z ∈ [0, 3]
    upper = [Pos(0, 25, 5.3) * Box(6, 50, 3), Pos(0, 0, 3.4) * Cylinder(2, 6.8)]  # plate z ∈ [3.8, 6.8] + hub
    if lug is not None:
        a = math.radians(lug)
        upper.append(Pos(40 * math.cos(a), 40 * math.sin(a), 3.4) * Box(2, 2, 6.8))
    asm.part("upper", upper)
    asm.revolute("j_lower", "base", "lower", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j_upper", "base", "upper", origin=(0, 0, 0), axis=(0, 0, 1))
    return asm


def test_co_hinged_layers_are_settled_by_their_profiles():
    """Two arms turning about one pivot (strandbeest's crank-side links): the lower arm sweeps 45°
    per frame under the upper one, 0.8 mm below it and 1 mm around its hub — the (height, radius)
    profiles about the pivot keep 0.8 mm at any angle, so no sub-frame pose is needed."""
    asm = _co_hinged(lug=None)
    checker = ClearanceChecker(asm)
    (pair,) = [p for p in checker._pairs if {p.a, p.b} == {"lower", "upper"}]
    assert not pair.excused and checker._hinge_line("lower", "upper") is not None
    assert 0.5 < checker._static_gap(pair) <= 0.8
    sweep = checker.sweep(run_study(asm, Kinematics(asm), Study("swing", {"j_lower": (0, 90)}, frames=3)))
    assert sweep.worst == {} and sweep.stats["subframe_poses"] == 0
    assert sweep.min_clearance[2] == pytest.approx(0.8, abs=1e-6)


def test_co_hinged_lug_in_the_sweep_is_found_between_frames():
    """The same arms with a block of the upper arm in the lower arm's path, between two frames:
    the profiles overlap, so the interval is bisected and the hit found (no shortcut hides it)."""
    asm = _co_hinged(lug=22.5)
    checker = ClearanceChecker(asm)
    (pair,) = [p for p in checker._pairs if {p.a, p.b} == {"lower", "upper"}]
    assert checker._static_gap(pair) == 0.0
    sweep = checker.sweep(run_study(asm, Kinematics(asm), Study("swing", {"j_lower": (0, 90)}, frames=3)))
    k, r = sweep.worst[frozenset(("lower", "upper"))]
    assert r.status == "interference" and 0 < r.at < 1 and k in (0, 1)


def test_check_clearance_exempts_the_carried_bore():
    """A5: check_clearance on a pinned pair holds it to clearance outside the pin-in-bore region —
    the 0.2 mm running gap of the bore is not a tight clearance — yet a lug face 0.1 mm from the
    eye is tight, and a pin too big for its bore still interferes."""
    from build123d import Rot

    def model(lug_gap: float, pin_r: float = 6.0) -> Assembly:
        asm = Assembly("clevis", clearance=0.3, pin_tol=8.0)
        stick = Pos(0, -20, 0) * Box(60, 8, 30) + Rot(90, 0, 0) * Cylinder(pin_r, 40)  # web + pin along y
        stick += Pos(0, 4 + lug_gap + 2, 0) * (Box(40, 4, 40) - Rot(90, 0, 0) * Cylinder(9, 10))  # a lug
        asm.part("stick", stick, ground=True)
        asm.part("barrel", Rot(90, 0, 0) * (Cylinder(10, 8) - Cylinder(6.2, 8)) + Pos(25, 0, 0) * Box(30, 8, 8))
        asm.revolute("j", "stick", "barrel", origin=(0, 0, 0), axis=(0, 1, 0))
        asm.check_clearance("stick", "barrel")
        return asm

    def sweep_of(asm: Assembly):
        return ClearanceChecker(asm).sweep(run_study(asm, Kinematics(asm), Study("s", {"j": (0, 30)}, frames=4)))

    free = sweep_of(model(lug_gap=1.0))
    assert free.worst == {}  # the 0.2 mm bore gap is exempt
    a, b, gap, _ = free.min_clearance
    assert {a, b} == {"stick", "barrel"} and gap == pytest.approx(1.0, abs=1e-6)  # the lug face
    _, r = sweep_of(model(lug_gap=0.1)).worst[frozenset(("stick", "barrel"))]
    assert r.status == "tight" and r.distance == pytest.approx(0.1, abs=1e-6)
    _, r = sweep_of(model(lug_gap=1.0, pin_r=6.5)).worst[frozenset(("stick", "barrel"))]
    assert r.status == "interference" and r.volume == pytest.approx(math.pi * (6.5**2 - 6.2**2) * 8, rel=1e-6)


def test_virtual_parts_have_no_material():
    asm = two_parts(CUBE10, cube10_at_gap(0.1))
    asm.part("knuckle", Pos(5, 5, 5) * Box(0.1, 0.1, 0.1))
    asm.revolute("j_k", "a", "knuckle", origin=(5, 5, 5), axis=(0, 0, 1))
    asm.parts["knuckle"].virtual = True
    pairs = ClearanceChecker(asm)._pairs
    assert pairs and all("knuckle" not in (p.a, p.b) for p in pairs)


def test_co_hinged_pair_settled_by_the_faces_within_reach():
    """A lug of the upper arm behind the pivot (180°) puts both arms' profiles at 40 mm radius in
    one layer, so no gap holds at every angle — but in 11.25° steps the lower arm never gets near
    it: the faces within reach keep their profiles apart and no interval is bisected."""
    asm = _co_hinged(lug=180.0)
    checker = ClearanceChecker(asm)
    (pair,) = [p for p in checker._pairs if {p.a, p.b} == {"lower", "upper"}]
    assert checker._static_gap(pair) == 0.0
    sweep = checker.sweep(run_study(asm, Kinematics(asm), Study("swing", {"j_lower": (0, 90)}, frames=9)))
    assert sweep.worst == {} and sweep.stats["subframe_poses"] == 0


def _piston_in_fins(lug: bool = False) -> Assembly:
    """A piston (r 10) on a prismatic joint along Z inside four fin rings (bore r 13): the cooling
    fins around an engine barrel, 3 mm off the piston however far it strokes; a stop 2 mm under
    the piston (off the slide line, so not a carrier of it) is the closest pair. With ``lug`` the piston carries a block reaching r 14 — into
    the fins' radius band, between two fins at home."""
    asm = Assembly("finned", clearance=0.5)
    asm.part("base", FAR, ground=True)
    asm.part("fins", [Pos(0, 0, z) * (Cylinder(20, 2) - Cylinder(13, 3)) for z in (0, 8, 16, 24)], ground=True)
    asm.part("stop", Pos(8, 0, -6) * Box(4, 4, 2), ground=True)  # x 6…10, top at z = −5
    piston = Pos(0, 0, 1) * Cylinder(10, 8)  # z ∈ [−3, 5]
    if lug:
        piston = piston + Pos(12, 0, 3) * Box(4, 2, 1)  # r 10…14, z 2.5…3.5: between the fins at 1 and 7
    asm.part("piston", piston)
    asm.prismatic("j", "base", "piston", origin=(0, 0, -4), axis=(0, 0, 1))
    return asm


def test_tess_radial_band_of_a_ring():
    from mech import tess

    mesh = tess.build_mesh((Cylinder(20, 2) - Cylinder(13, 3)).wrapped, 0.01, 0.2)
    lo, hi = tess.radial_band(np.concatenate(mesh.face_tris), (0, 0, 0), (0, 0, 1))
    assert lo == pytest.approx(13.0, abs=0.02) and hi == pytest.approx(20.0, abs=1e-9)
    lo, hi = tess.radial_band(np.concatenate(mesh.face_tris), (30, 0, 0), (0, 0, 1))  # a line beside it
    assert lo == pytest.approx(10.0, abs=0.02) and hi == pytest.approx(50.0, abs=0.02)


def test_slide_line_keeps_a_piston_clear_of_its_fins_without_measuring_them():
    """A piston sliding inside the fins around its barrel: its radius band about the slide line
    (0…10) and the fins' (13…20) are 3 mm apart at every stroke, so the pair — the boxes overlap
    every frame — is never measured once the stop sets a closer minimum."""
    asm = _piston_in_fins()
    checker = ClearanceChecker(asm)
    (pair,) = [p for p in checker._pairs if {p.a, p.b} == {"fins", "piston"}]
    assert checker._hinge_line("fins", "piston") is None and checker._slide_line("fins", "piston") is not None
    assert 2.9 < checker._slide_gap(pair) <= 3.0
    sweep = checker.sweep(run_study(asm, Kinematics(asm), Study("stroke", {"j": (0, 20)}, frames=5)))
    assert sweep.worst == {}
    assert sweep.min_clearance[:3] == ("stop", "piston", pytest.approx(2.0, abs=1e-6))
    assert ("fins", "piston") not in checker._pair_time  # never queried: the band settled it
    # alone, the pair is the closest one: then it is measured, exactly
    lone = ClearanceChecker(asm)
    lone._pairs = [p for p in lone._pairs if {p.a, p.b} == {"fins", "piston"}]
    sweep = lone.sweep(run_study(asm, Kinematics(asm), Study("stroke", {"j": (0, 20)}, frames=5)))
    assert sweep.min_clearance[2] == pytest.approx(3.0, abs=1e-6)


def test_slide_line_bound_needs_disjoint_bands_and_a_pure_slide():
    """A lug reaching into the fins' band voids the bound (and the hit is found between frames);
    the bound applies only to poses that slide along or turn about the line."""
    from mech.clearance import _on_line

    asm = _piston_in_fins(lug=True)
    checker = ClearanceChecker(asm)
    (pair,) = [p for p in checker._pairs if {p.a, p.b} == {"fins", "piston"}]
    assert checker._slide_gap(pair) == 0.0
    sweep = checker.sweep(run_study(asm, Kinematics(asm), Study("stroke", {"j": (0, 6)}, frames=2)))
    k, r = sweep.worst[frozenset(("fins", "piston"))]
    assert r.status == "interference"  # the lug (z 2.5…3.5) meets the fin at z 7…9 on the way up
    lp, lu = np.array([[0.0, 0.0, -4.0]]), np.array([[0.0, 0.0, 1.0]])
    poses = np.stack([translation((0, 0, 7)), rot_about_line((0, 0, 3), (0, 0, 1), 40) @ translation((0, 0, -2)),
                      translation((0.01, 0, 7)), rot_about_line((1, 0, 0), (0, 0, 1), 40)])
    assert list(_on_line(poses, np.repeat(lp, 4, 0), np.repeat(lu, 4, 0), slide=True)) == [True, True, False, False]
    # without ``slide`` only a turn about the line itself qualifies (the hinge-profile bound's rule)
    assert list(_on_line(poses, np.repeat(lp, 4, 0), np.repeat(lu, 4, 0))) == [False, False, False, False]


def test_surface_gap_of_planes_cylinders_and_spheres():
    """Lower bounds between infinite surfaces: a face lies on its surface, so faces are at least as
    far apart; surfaces that may meet (non-parallel planes / axes) give 0."""
    from mech.tess import _pair_gap

    z, x = np.array([0.0, 0, 1]), np.array([1.0, 0, 0])
    o = np.zeros(3)
    ball = ("sphere", np.array([0.0, 0.1, 0]), 5.0)
    assert _pair_gap(("cylinder", o, z, 5.15), ball) == pytest.approx(0.05)    # ball off-centre in its ring
    assert _pair_gap(ball, ("plane", np.array([0.0, 0, 7]), z)) == pytest.approx(2.0)
    assert _pair_gap(ball, ("sphere", o, 5.5)) == pytest.approx(0.4)            # nested spheres
    assert _pair_gap(("cylinder", np.array([0.0, 0.2, 0]), z, 6.0), ("cylinder", o, z, 6.4)) == pytest.approx(0.2)
    assert _pair_gap(("cylinder", o, z, 6.0), ("cylinder", np.array([20.0, 0, 0]), z, 6.0)) == pytest.approx(8.0)
    assert _pair_gap(("cylinder", o, z, 6.0), ("cylinder", o, x, 6.4)) == 0.0     # crossing axes
    assert _pair_gap(("plane", o, z), ("plane", np.array([0, 0, 3.0]), -z)) == pytest.approx(3.0)
    assert _pair_gap(("plane", o, z), ("plane", o, x)) == 0.0
    assert _pair_gap(("cylinder", np.array([0, 0, 9.0]), x, 4.0), ("plane", o, z)) == pytest.approx(5.0)


def _ball_in_ring(bore_r: float) -> Assembly:
    """A ø10 ball stud swinging in a ring of bore radius ``bore_r`` (a rod-end socket), closed by a
    ball pin at the ball centre: a joined pair whose only near faces are a sphere and a cylinder."""
    asm = Assembly("socket", clearance=0.5)
    asm.part("base", Pos(0, 0, -15) * Box(30, 30, 4), ground=True)
    asm.part("stud", Sphere(5) + Pos(0, 0, -6.5) * Cylinder(1.5, 13), ground=True)  # neck down to the base
    asm.part("rod", Cylinder(8, 6) - Cylinder(bore_r, 8) + Pos(20, 0, 0) * Box(26, 6, 6))
    asm.revolute("j", "base", "rod", origin=(0, 0, 0), axis=(0, 1, 0))
    asm.pin("p", "rod", "stud", point=(0, 0, 0))
    return asm


@pytest.mark.parametrize("bore_r, gap", [(5.15, True), (5.0, False)])
def test_ball_in_its_socket_is_proven_clear_by_its_surfaces(bore_r, gap):
    """0.15 mm of play around the ball is inside the meshes' error band, so the proximity test
    can't prove it — the sphere and the bore cylinder do, with no exact distance query. A ball
    on its bore (no play) still gets the exact query, and touches."""
    asm = _ball_in_ring(bore_r)
    assert asm.is_joined("rod", "stud")
    checker = ClearanceChecker(asm)
    exact = Counter()  # exact queries per pair (the rod also swings past the base: guarded apart)
    measure = checker._exact

    def counted(pair, M, near, stats):
        exact[frozenset((pair.a, pair.b))] += 1
        return measure(pair, M, near, stats)

    checker._exact = counted
    sweep = checker.sweep(run_study(asm, Kinematics(asm), Study("swing", {"j": (-30, 30)}, frames=5)))
    socket = frozenset(("rod", "stud"))
    if gap:  # clear at every frame, proven without one exact query
        assert sweep.worst == {}
        assert sweep.stats["surface_bounds"] >= 1 and exact[socket] == 0
    else:  # touching (joined: contact, never tight nor interference), measured exactly
        assert {r.status for _, r in sweep.worst.values()} == {"contact"}
        assert sweep.stats.get("surface_bounds", 0) == 0 and exact[socket] >= 1


# ------------------------------------------------------------------------------ review round 3


def _buried_pin(hinged: bool) -> Assembly:
    """A ø5 pin turning in a 10 mm plate that was never bored: on a revolute to the plate (an
    excused pair), or co-hinged with it through a third body (a pair held to clearance)."""
    asm = Assembly("buried", clearance=0.3)
    asm.part("plate", Box(40, 40, 10), ground=True)  # z ∈ [−5, 5], no bore
    asm.part("pin", Cylinder(2.5, 8))  # z ∈ [−4, 4]: inside the plate
    if hinged:
        asm.revolute("j", "plate", "pin", origin=(0, 0, 0), axis=(0, 0, 1))
    else:
        asm.part("hub", Pos(0, 0, -500) * Box(4, 4, 4))
        asm.revolute("j", "plate", "hub", origin=(0, 0, -500), axis=(0, 0, 1))
        asm.revolute("j2", "hub", "pin", origin=(0, 0, -500), axis=(0, 0, 1))
    return asm


@pytest.mark.parametrize("hinged", [True, False])
def test_a_part_buried_on_its_hinge_line_interferes(hinged):
    """The (height, radius) profile bound sees boundaries only: a pin buried in an unbored plate
    has a profile apart from the plate's, yet overlaps it at every angle. Containment holds along
    the whole turn, so the bound is 0 and the pin's whole volume interferes — excused or not."""
    asm = _buried_pin(hinged)
    checker = ClearanceChecker(asm)
    (pair,) = [p for p in checker._pairs if {p.a, p.b} == {"plate", "pin"}]
    assert pair.excused == hinged and checker._hinge_line("plate", "pin") is not None
    assert checker._static_gap(pair) == 0.0
    (r,) = [r for r in checker.check_pose({}) if {r.a, r.b} == {"plate", "pin"}]
    assert r.status == "interference" and r.volume == pytest.approx(math.pi * 2.5**2 * 8, rel=1e-6)


def test_buried_co_hinged_part_is_found_past_a_closer_pair():
    """A stub buried in a disc, both on revolutes about one line from a third body, plus an
    unrelated pair 1 mm apart: the closest pair found first must not let the buried pair's
    profile bound (1.8 mm) skip it."""
    asm = Assembly("cohinge", clearance=0.3)
    asm.part("base", Pos(0, 0, -50) * Box(10, 10, 4), ground=True)
    asm.part("disc", Pos(0, 0, 5) * Cylinder(40, 10))
    asm.part("stub", Pos(30, 0, 5) * Cylinder(3, 6))  # z ∈ [2, 8] inside the disc
    asm.part("post", Pos(-60, 0, 5) * Box(6, 6, 10), ground=True)
    asm.part("arm2", Pos(-47.5, 0, 5) * Box(13, 4, 4))  # 1 mm outside the disc
    for j, child in (("j1", "disc"), ("j2", "stub"), ("j3", "arm2")):
        asm.revolute(j, "base", child, origin=(0, 0, 0), axis=(0, 0, 1))
    assert not asm.is_joined("disc", "stub")
    sweep = ClearanceChecker(asm).sweep(
        run_study(asm, Kinematics(asm), Study("s", {"j1": (0, 30), "j2": (0, 30)}, frames=4)))
    _, r = sweep.worst[frozenset(("disc", "stub"))]
    assert r.status == "interference" and r.volume == pytest.approx(math.pi * 9 * 6, rel=1e-3)
    assert set(sweep.min_clearance[:2]) == {"disc", "stub"} and sweep.min_clearance[2] < 0


def test_gear_in_a_housing_never_hollowed_out_fails():
    """A spur gear on a revolute inside a solid block (the cavity forgotten)."""
    gp = gear_pair(1.0, 15, 30, 6, bore1=5, bore2=5)
    asm = Assembly("housing", clearance=0.3)
    asm.part("housing", Box(60, 60, 20), ground=True)
    asm.part("gear", gp.g1)
    asm.revolute("j", "housing", "gear", origin=(0, 0, 0), axis=(0, 0, 1))
    sweep = ClearanceChecker(asm).sweep(run_study(asm, Kinematics(asm), Study("turn", {"j": (0, 360)}, frames=5)))
    _, r = sweep.worst[frozenset(("housing", "gear"))]
    assert r.status == "interference" and r.volume > 800


def _rotor_in_bore(boss_deg: float | None = None, frames: int = 12) -> Assembly:
    """A rotor arm (hub r 3, tip at r 18) turning in a housing's ø40 bore, carried by a frame
    revolute; with ``boss_deg`` the housing has a boss reaching in to r 16.5 at that angle."""
    from build123d import Rot

    asm = Assembly("bore", clearance=0.3)
    housing = Pos(0, 0, 5) * (Box(60, 60, 10) - Cylinder(20, 10))
    if boss_deg is not None:
        a = math.radians(boss_deg)
        housing += Pos(18.5 * math.cos(a), 18.5 * math.sin(a), 5) * Rot(0, 0, boss_deg) * Box(4, 3, 10)
    asm.part("frame", Pos(0, 0, -30) * Box(10, 10, 4), ground=True)
    asm.part("housing", housing, ground=True)
    asm.part("rotor", Pos(0, 0, 5) * Cylinder(3, 10) + Pos(9, 0, 5) * Box(18, 4, 4))
    asm.revolute("j", "frame", "rotor", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.study("turn", drive={"j": (0, 360)}, frames=frames)
    return asm


def test_an_arm_turning_in_a_big_bore_is_not_carried_by_it():
    """A3 joins a part around the hinge axis with what runs in its bore — a journal filling the
    bore, not an arm whose tip alone nears the wall: the housing/rotor pair keeps clearance and
    the between-frame guard, so a boss passed between two frames is found."""
    asm = _rotor_in_bore()
    assert not asm.is_joined("housing", "rotor")
    asm = _rotor_in_bore(boss_deg=81.8)
    sweep = ClearanceChecker(asm).sweep(run_study(asm, Kinematics(asm), asm.studies[0]))
    _, r = sweep.worst[frozenset(("housing", "rotor"))]
    assert r.status == "interference" and r.at is not None and 2 < r.at < 3


def _slider_crank_stop(*, crank_home: float = 0.0, stop_x: float, lug: bool = False) -> Assembly:
    """A slider-crank (r 30, l 90) with a ground stop beyond the slider's dead centre: on the slide
    line, or (``lug``) off it, meeting a lug on the slider; the crank drawn at ``crank_home``°."""
    from mech.geom import link

    r, l, t = 30.0, 90.0, 5.0
    a = math.radians(crank_home)
    A = (r * math.cos(a), r * math.sin(a), 0.0)
    xh = A[0] + math.sqrt(l * l - A[1] ** 2)
    C = (xh, 0.0, 0.0)
    asm = Assembly("tdc", clearance=0.3)
    asm.part("frame", Pos(60, 0, -t) * Box(200, 20, t), ground=True)
    asm.part("stop", Pos(stop_x + 5, 13 if lug else 0, 1.25) * Box(10, 10 if lug else 12, 7.5), ground=True)
    asm.part("crank", link((0, 0, 0), A, width=10, thickness=t, z=0))
    asm.part("rod", link(A, C, width=10, thickness=t, z=t + 0.5))
    slider = [Pos(xh, 0, t / 4) * Box(20, 12, 1.5 * t)]
    if lug:
        slider.append(Pos(xh + 5, 11, t / 4) * Box(10, 10, 1.5 * t))
    asm.part("slider", slider)
    asm.revolute("j_crank", "frame", "crank", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j_rod", "crank", "rod", origin=A, axis=(0, 0, 1))
    asm.prismatic("j_slider", "frame", "slider", origin=C, axis=(1, 0, 0), home=xh)
    asm.pin("p_C", "rod", "slider", point=C, axis=(0, 0, 1))
    return asm


def test_a_reversal_between_two_frames_is_not_settled_by_its_zero_chord():
    """The slider's dead centre falls midway between two frames, so its pose relative to a stop is
    the same at both — a chord of 0. The path bends there (the neighbouring frames show it), so
    the interval is bisected and the lug's hit on the stop at dead centre is found."""
    frames = 12
    a = math.radians(360 / (frames - 1) / 2)
    xs = 30 * math.cos(a) + math.sqrt(90**2 - (30 * math.sin(a)) ** 2)  # slider x at the frames by TDC
    asm = _slider_crank_stop(crank_home=90.0, stop_x=xs + 10 + 0.8, lug=True)
    assert not asm.is_joined("stop", "slider")
    res = run_study(asm, Kinematics(asm), Study("turn", {"j_crank": (-270, 90)}, frames=frames))
    _, r = ClearanceChecker(asm).sweep(res).worst[frozenset(("stop", "slider"))]
    assert r.status == "interference" and r.at == pytest.approx(5.5, abs=0.05)


def test_a_stop_on_the_slide_line_is_not_carried_by_the_slide():
    """A prismatic joint's line runs on past the slider: a stop on it 0.15 mm beyond the slider's
    dead centre does not carry the slider (only parts side by side along the line do — a guide
    rod in its bushing), so the near miss at dead centre, between two frames, is tight."""
    asm = _slider_crank_stop(stop_x=130.15)
    assert not asm.is_joined("stop", "slider")
    res = run_study(asm, Kinematics(asm), Study("turn", {"j_crank": (-180, 180)}, frames=12))
    _, r = ClearanceChecker(asm).sweep(res).worst[frozenset(("stop", "slider"))]
    assert r.status == "tight" and r.distance == pytest.approx(0.15, abs=1e-3) and r.at is not None


@pytest.mark.parametrize("distance, status", [(0.3, "ok"), (0.29999999999999716, "ok"), (0.2999, "tight")])
def test_a_gap_built_to_the_clearance_is_not_tight(distance, status):
    """A gap built exactly to the required clearance lands on it up to float noise (0.3 − 3e-15):
    with the targets' allowance it meets the clearance, as a clearance target says it does."""
    asm = two_parts(CUBE10, cube10_at_gap(1.0))
    checker = ClearanceChecker(asm)
    (pair,) = [p for p in checker._pairs if {p.a, p.b} == {"a", "b"}]
    assert checker._status(pair, SimpleNamespace(volume=0.0, touching=False, distance=distance)) == status


def test_unresolved_between_frame_intervals_are_counted_per_pair():
    """Two lids hinged on opposite edges, 0.2 mm apart at home, opening: the sub-frame check gives
    up on the first intervals (tight at one end while the far edges sweep fast) — the stats say how
    many, and for which pair."""
    asm = Assembly("lids", clearance=0.3)
    asm.part("body", Pos(0, 0, -10) * Box(60, 40, 4), ground=True)
    asm.part("lid_left", Pos(-15.1, 0, 1) * Box(29.8, 40, 2))
    asm.part("lid_right", Pos(15.1, 0, 1) * Box(29.8, 40, 2))
    asm.revolute("j_left", "body", "lid_left", origin=(-30, 0, 0), axis=(0, -1, 0))
    asm.revolute("j_right", "body", "lid_right", origin=(30, 0, 0), axis=(0, 1, 0))
    res = run_study(asm, Kinematics(asm), Study("open", {"j_left": (0, 110), "j_right": (0, 110)}, frames=12))
    st = ClearanceChecker(asm).sweep(res).stats
    assert st["sub_unresolved"] >= 1
    assert st["unresolved_pairs"] == [["lid_left", "lid_right", st["sub_unresolved"]]]


def test_check_clearance_exempts_a_pin_in_an_eye_bigger_than_pin_tol():
    """A5 for real eyes: a ø12 pin in a ø12.8 eye (bore radius 6.4 > pin_tol 5) under
    check_clearance — the 0.4 mm radial bore gap stays exempt, the cheek faces 1 mm from the eye
    are measured (and 0.3 mm cheeks are tight)."""
    from build123d import Rot

    def ycyl(r, y0, y1):
        return Pos(0, (y0 + y1) / 2, 0) * Rot(90, 0, 0) * Cylinder(r, y1 - y0)

    def model(cheek: float) -> Assembly:
        asm = Assembly("cc", clearance=0.5)
        clevis = Pos(-20, 0, 0) * Box(10, 30, 30) + ycyl(6, -15, 15)
        for s in (-1, 1):
            clevis += Pos(0, s * (5 + cheek + 2.5), 0) * Box(30, 5, 30)
        asm.part("clevis", clevis, ground=True)
        asm.part("arm", ycyl(12, -5, 5) - ycyl(6.4, -6, 6) + Pos(40, 0, 0) * Box(60, 10, 10))
        asm.revolute("j", "clevis", "arm", origin=(0, 0, 0), axis=(0, 1, 0))
        asm.check_clearance("arm", "clevis")
        return asm

    def sweep_of(asm):
        return ClearanceChecker(asm).sweep(run_study(asm, Kinematics(asm), Study("s", {"j": (0, 20)}, frames=3)))

    free = sweep_of(model(1.0))
    assert free.worst == {} and free.min_clearance[2] == pytest.approx(1.0, abs=1e-6)
    _, r = sweep_of(model(0.3)).worst[frozenset(("clevis", "arm"))]
    assert r.status == "tight" and r.distance == pytest.approx(0.3, abs=1e-6)


def test_a_joined_pair_apart_at_both_frames_is_guarded_for_overlap():
    """m_tdc: a stop drawn overlapping the slider at dead centre (home) touches it at home, so the
    pair is joined; at the frames either side of dead centre it is 0.7 mm clear. Joined pairs are
    checked for overlap only, but where both ends of an interval are proven apart they are guarded
    between frames too — the overlap at dead centre, between f5 and f6, is found."""
    asm = _slider_crank_stop(stop_x=129.19)
    assert asm.is_joined("stop", "slider")
    res = run_study(asm, Kinematics(asm), Study("turn", {"j_crank": (-180, 180)}, frames=12))
    _, r = ClearanceChecker(asm).sweep(res).worst[frozenset(("stop", "slider"))]
    assert r.status == "interference" and r.at == pytest.approx(5.5, abs=0.05)
    assert r.volume == pytest.approx(0.81 * 12 * 7.5, rel=0.05)
