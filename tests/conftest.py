"""Shared test fixtures: programmatic reference mechanisms with closed-form kinematics."""

from __future__ import annotations

import math

import pytest
from build123d import Box, Cylinder, Pos

from mech import Assembly
from mech.geom import circle_intersect, link


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run tests marked slow (full showcase analyses; same as -m slow)")


def pytest_collection_modifyitems(config, items):
    """``slow`` tests are skipped unless ``--runslow`` is given or ``-m`` selects them."""
    if config.getoption("--runslow") or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow or -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

FOUR_BAR = dict(ground=100.0, crank=40.0, coupler=90.0, rocker=80.0, t=5.0)
FOUR_BAR_LINKS = {k: v for k, v in FOUR_BAR.items() if k != "t"}  # link lengths only
SLIDER_CRANK = dict(r=30.0, l=90.0, t=5.0)


def make_four_bar(ground=100.0, crank=40.0, coupler=90.0, rocker=80.0, t=5.0) -> Assembly:
    """The §2 crank-rocker, built with geom.link + circle_intersect (crank drawn at 0°)."""
    asm = Assembly("four_bar", clearance=0.3)
    O2, O4 = (0, 0, 0), (ground, 0, 0)
    A = (crank, 0, 0)
    B = circle_intersect(A, coupler, O4, rocker, side=+1)
    asm.part("frame", Pos(ground / 2, 0, -2 * t) * Box(ground + 20, 16, t), ground=True, color="#666666")
    asm.part("crank", link(O2, A, width=10, thickness=t, z=0), material="aluminum_6061")
    asm.part("coupler", link(A, B, width=10, thickness=t, z=t + 0.5), material="aluminum_6061")
    asm.part("rocker", link(O4, B, width=10, thickness=t, z=2 * t + 1), material="aluminum_6061")
    asm.revolute("j_crank", "frame", "crank", origin=O2, axis=(0, 0, 1))
    asm.revolute("j_coupler", "crank", "coupler", origin=A, axis=(0, 0, 1))
    asm.revolute("j_rocker", "frame", "rocker", origin=O4, axis=(0, 0, 1))
    asm.pin("p_B", "coupler", "rocker", point=B, axis=(0, 0, 1))
    asm.probe("mid", part="coupler", point=[(a + b) / 2 for a, b in zip(A, B)])
    asm.actuator("j_crank", capacity=0.5)
    asm.study("turn", drive={"j_crank": (0, 360)}, frames=72)
    asm.target("rocker swing", "span:j_rocker", min=40)
    return asm


def four_bar_rocker_deg(theta2_deg: float, ground=100.0, crank=40.0, coupler=90.0, rocker=80.0) -> float:
    """Absolute rocker angle θ4 (deg) on the drawn (open) branch, from Freudenstein's equation

        K1·cos θ4 − K2·cos θ2 + K3 = cos(θ2 − θ4),  K1 = d/a, K2 = d/c, K3 = (a² − b² + c² + d²)/(2ac),

    rearranged to P·cos θ4 + Q·sin θ4 = R with P = K1 − cos θ2, Q = −sin θ2, R = K2·cos θ2 − K3.
    The '+' root is the drawn branch (B above O2→O4). This Grashof crank-rocker never folds the
    coupler onto the rocker, so the branch holds over a full crank turn.
    """
    a, b, c, d = crank, coupler, rocker, ground
    t2 = math.radians(theta2_deg)
    k1, k2, k3 = d / a, d / c, (a * a - b * b + c * c + d * d) / (2 * a * c)
    P, Q, R = k1 - math.cos(t2), -math.sin(t2), k2 * math.cos(t2) - k3
    return math.degrees(math.atan2(Q, P) + math.acos(R / math.hypot(P, Q)))


def make_slider_crank(r=30.0, l=90.0, t=5.0) -> Assembly:
    """Zero-offset slider-crank: crank r drawn at 0°, rod l, slider home = r + l (so x = q_slider)."""
    asm = Assembly("slider_crank")
    O, A, C = (0, 0, 0), (r, 0, 0), (r + l, 0, 0)
    asm.part("frame", Pos((r + l) / 2, 0, -t) * Box(r + l + 80, 20, t), ground=True)
    asm.part("crank", link(O, A, width=10, thickness=t, z=0))
    asm.part("rod", link(A, C, width=10, thickness=t, z=t + 0.5))
    asm.part("slider", Pos(r + l, 0, 1.25) * Box(20, 12, 7.5))
    asm.revolute("j_crank", "frame", "crank", origin=O, axis=(0, 0, 1))
    asm.revolute("j_rod", "crank", "rod", origin=A, axis=(0, 0, 1))
    asm.prismatic("j_slider", "frame", "slider", origin=C, axis=(1, 0, 0), home=r + l)
    asm.pin("p_C", "rod", "slider", point=C, axis=(0, 0, 1))
    return asm


def make_disc(center, r=10.0, h=4.0):
    """A disc of radius r whose bottom face sits at center.z (axis +Z)."""
    x, y, z = center
    return Pos(x, y, z + h / 2) * Cylinder(r, h)


@pytest.fixture
def four_bar() -> Assembly:
    return make_four_bar(**FOUR_BAR)


@pytest.fixture
def slider_crank() -> Assembly:
    return make_slider_crank(**SLIDER_CRANK)
