"""ISO 286 limits and fits (nominal sizes up to 120 mm), plus practical hole-size helpers.

`fit(10, "H7/g6")` combines the ISO 286-1 standard tolerance grades (IT table) with the fundamental
deviations of ISO 286-2 for the tolerance-zone letters below. Values are the published table values
in micrometres for each nominal-size range ("over a, up to and including b").

Supported zones: shafts c d e f g h js k m n p, holes C D E F G H JS (hole deviations for A..H are
the shaft ones mirrored, EI = −es, with no Δ correction). Grades IT1..IT13.
"""
from __future__ import annotations

import math
import re
from bisect import bisect_left

from ._build import lookup_metric, metric_size

# Upper bounds (mm) of the ISO 286 nominal-size ranges up to 120 mm: (0,3], (3,6], ..., (80,120].
_RANGES = (3, 6, 10, 18, 30, 50, 80, 120)

# ISO 286-1 Table 1: standard tolerance grades, µm, one entry per range above.
_IT: dict[int, tuple[float, ...]] = {
    1: (0.8, 1, 1, 1.2, 1.5, 1.5, 2, 2.5),
    2: (1.2, 1.5, 1.5, 2, 2.5, 2.5, 3, 4),
    3: (2, 2.5, 2.5, 3, 4, 4, 5, 6),
    4: (3, 4, 4, 5, 6, 7, 8, 10),
    5: (4, 5, 6, 8, 9, 11, 13, 15),
    6: (6, 8, 9, 11, 13, 16, 19, 22),
    7: (10, 12, 15, 18, 21, 25, 30, 35),
    8: (14, 18, 22, 27, 33, 39, 46, 54),
    9: (25, 30, 36, 43, 52, 62, 74, 87),
    10: (40, 48, 58, 70, 84, 100, 120, 140),
    11: (60, 75, 90, 110, 130, 160, 190, 220),
    12: (100, 120, 150, 180, 210, 250, 300, 350),
    13: (140, 180, 220, 270, 330, 390, 460, 540),
}

# ISO 286-2 upper deviations es (µm) of shafts a..h. Letter c changes at the intermediate
# steps 40, 65 and 100 mm, so it is stored per (range, sub-range).
_ES: dict[str, tuple[float, ...]] = {
    "d": (-20, -30, -40, -50, -65, -80, -100, -120),
    "e": (-14, -20, -25, -32, -40, -50, -60, -72),
    "f": (-6, -10, -13, -16, -20, -25, -30, -36),
    "g": (-2, -4, -5, -6, -7, -9, -10, -12),
    "h": (0, 0, 0, 0, 0, 0, 0, 0),
}
_C_ES = ((3, -60), (6, -70), (10, -80), (18, -95), (30, -110), (40, -120), (50, -130),
         (65, -140), (80, -150), (100, -170), (120, -180))

# ISO 286-2 lower deviations ei (µm) of shafts k..p; k applies to IT4..IT7 only (0 otherwise).
_EI: dict[str, tuple[float, ...]] = {
    "k": (0, 1, 1, 1, 2, 2, 2, 3),
    "m": (2, 4, 6, 7, 8, 9, 11, 13),
    "n": (4, 8, 10, 12, 15, 17, 20, 23),
    "p": (6, 12, 15, 18, 22, 26, 32, 37),
}

_ZONE = re.compile(r"^([a-z]{1,2}|[A-Z]{1,2})(\d{1,2})$")


def _range_index(nominal: float) -> int:
    if not 0 < nominal <= _RANGES[-1]:
        raise ValueError(f"ISO 286 tables here cover nominal sizes 0 < d <= 120 mm, got {nominal}")
    return bisect_left(_RANGES, nominal)      # "over a up to and including b"


def it_grade(nominal: float, grade: int) -> float:
    """Standard tolerance ITgrade in µm for the nominal size (mm)."""
    if grade not in _IT:
        raise ValueError(f"IT grade must be one of {sorted(_IT)}, got {grade}")
    return _IT[grade][_range_index(nominal)]


def _shaft_deviations(nominal: float, letter: str, grade: int) -> tuple[float, float]:
    """(ei, es) in µm for a shaft zone such as g6."""
    i = _range_index(nominal)
    it = it_grade(nominal, grade)
    if letter == "js":
        # ±IT/2; for grades 7..11 an odd IT value (µm) is first rounded down to the next even value.
        if 7 <= grade <= 11 and float(it).is_integer() and int(it) % 2:
            it -= 1
        return -it / 2, it / 2
    if letter == "c":
        es = next(v for upper, v in _C_ES if nominal <= upper)
        return es - it, es
    if letter in _ES:
        es = _ES[letter][i]
        return es - it, es
    if letter in _EI:
        ei = _EI[letter][i]
        if letter == "k" and not 4 <= grade <= 7:
            ei = 0
        return ei, ei + it
    raise ValueError(f"unsupported shaft tolerance letter {letter!r}")


def tolerance_zone(nominal: float, zone: str) -> tuple[float, float]:
    """(min, max) size in mm of a feature toleranced as `zone`, e.g. ("H7" hole, "g6" shaft)."""
    match = _ZONE.match(zone)
    if not match:
        raise ValueError(f"bad tolerance zone {zone!r} (expected e.g. 'H7' or 'g6')")
    letters, grade = match.group(1), int(match.group(2))
    if letters.islower():
        lo, hi = _shaft_deviations(nominal, letters, grade)
    else:
        if letters.lower() not in ("c", "d", "e", "f", "g", "h", "js"):
            raise ValueError(f"unsupported hole tolerance letter {letters!r} (C D E F G H JS)")
        ei, es = _shaft_deviations(nominal, letters.lower(), grade)
        lo, hi = -es, -ei                    # EI = −es, ES = −ei (mirror rule for A..H, JS)
    return round(nominal + lo / 1000, 6), round(nominal + hi / 1000, 6)


def fit(nominal: float, spec: str) -> dict[str, tuple[float, float]]:
    """ISO 286 fit, e.g. `fit(10, "H7/g6")`.

    Returns sizes in mm: `hole` (min, max), `shaft` (min, max) and `clearance` (min, max), where
    clearance = hole − shaft and a negative value is interference.
    """
    try:
        hole_zone, shaft_zone = spec.replace(" ", "").split("/")
    except ValueError:
        raise ValueError(f"bad fit {spec!r} (expected e.g. 'H7/g6')") from None
    if not (hole_zone[:1].isupper() and shaft_zone[:1].islower()):
        raise ValueError(f"bad fit {spec!r}: hole zone first (upper case), then shaft (lower case)")
    hole = tolerance_zone(nominal, hole_zone)
    shaft = tolerance_zone(nominal, shaft_zone)
    clearance = (round(hole[0] - shaft[1], 6), round(hole[1] - shaft[0], 6))
    return {"hole": hole, "shaft": shaft, "clearance": clearance}


# --- practical hole sizes -------------------------------------------------------------------------

# ISO 273 clearance holes (mm): (fine, medium, coarse).
_ISO273 = {
    2: (2.2, 2.4, 2.6), 2.5: (2.7, 2.9, 3.1), 3: (3.2, 3.4, 3.6), 4: (4.3, 4.5, 4.8),
    5: (5.3, 5.5, 5.8), 6: (6.4, 6.6, 7.0), 8: (8.4, 9.0, 10.0), 10: (10.5, 11.0, 12.0),
    12: (13.0, 13.5, 14.5),
}
_ISO273_SERIES = {"close": 0, "fine": 0, "normal": 1, "medium": 1, "loose": 2, "coarse": 2}

# ISO 261 coarse pitches (mm).
COARSE_PITCH = {2: 0.4, 2.5: 0.45, 3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75}

# Typical brass heat-set inserts for 3D prints (Ruthex-style): hole dia, insert OD, insert length.
_HEAT_SET = {
    2: (3.2, 3.6, 3.0), 2.5: (4.0, 4.6, 4.0), 3: (4.0, 4.6, 5.7), 4: (5.6, 6.3, 8.1),
    5: (6.4, 7.1, 9.5), 6: (8.0, 8.7, 12.7),
}


def fdm_hole(d: float, clearance: float = 0.2) -> float:
    """Diameter to model so an FDM-printed hole fits a `d` pin: d plus a diametral allowance."""
    return d + clearance


def clearance_hole(size: str | float, fit: str = "normal") -> float:
    """ISO 273 clearance hole for a metric bolt; fit = close|normal|loose (fine|medium|coarse)."""
    if fit not in _ISO273_SERIES:
        raise ValueError(f"fit must be one of {sorted(_ISO273_SERIES)}, got {fit!r}")
    return lookup_metric(_ISO273, size, "clearance_hole")[_ISO273_SERIES[fit]]


def tap_drill(size: str | float) -> float:
    """Tap drill for an ISO coarse thread: nominal diameter − pitch (e.g. M3 -> 2.5)."""
    return round(metric_size(size) - lookup_metric(COARSE_PITCH, size, "tap_drill"), 3)


def heat_set_insert(size: str | float) -> dict[str, float]:
    """Typical heat-set insert: {"hole_d", "outer_d", "length"} in mm (hole_d = hole to model)."""
    hole_d, outer_d, length = lookup_metric(_HEAT_SET, size, "heat_set_insert")
    return {"hole_d": hole_d, "outer_d": outer_d, "length": length}
