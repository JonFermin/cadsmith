"""Metric fasteners M2–M8 with standard dimensions (simplified: no threads, sharp edges).

Screws: axis +Z, head underside at z=0, shank toward −Z; frames `"head"` (z=0) and `"tip"`
(z=−length), both +Z, so `screw.mate("head", surface_frame)` seats the head on a surface whose
frame Z points out of the material. Nuts and washers sit on z=0 and extend toward +Z; frames
`"bottom"` and `"top"`. Masses are steel (7.85 g/cm³) from the modeled volume.
"""
from __future__ import annotations

from collections.abc import Sequence

from build123d import Shape, Solid

from ._build import (STEEL, assemble, disk, lookup_metric, mass_g, metric_size, prism,
                     regular_polygon, tube, z_frame)
from .libpart import LibPart

# ISO 4762 socket head cap screws: head dia dk, head height k, hex key s, socket depth t (min).
_ISO4762 = {
    2: (3.8, 2.0, 1.5, 1.0), 2.5: (4.5, 2.5, 2.0, 1.1), 3: (5.5, 3.0, 2.5, 1.3),
    4: (7.0, 4.0, 3.0, 2.0), 5: (8.5, 5.0, 4.0, 2.5), 6: (10.0, 6.0, 5.0, 3.0),
    8: (13.0, 8.0, 6.0, 4.0),
}
# ISO 7380-1 button head screws: head dia dk, head height k, hex key s, socket depth t (min).
_ISO7380 = {
    3: (5.7, 1.65, 2.0, 1.04), 4: (7.6, 2.2, 2.5, 1.3), 5: (9.5, 2.75, 3.0, 1.56),
    6: (10.5, 3.3, 4.0, 2.08), 8: (14.0, 4.4, 5.0, 2.6),
}
# ISO 4032 hex nuts (style 1): width across flats s, height m.
_ISO4032 = {
    2: (4.0, 1.6), 2.5: (5.0, 2.0), 3: (5.5, 2.4), 4: (7.0, 3.2), 5: (8.0, 4.7), 6: (10.0, 5.2),
    8: (13.0, 6.8),
}
# ISO 7089 plain washers (normal series): inner d1, outer d2, thickness h.
_ISO7089 = {
    2: (2.2, 5.0, 0.3), 2.5: (2.7, 6.0, 0.5), 3: (3.2, 7.0, 0.5), 4: (4.3, 9.0, 0.8),
    5: (5.3, 10.0, 1.0), 6: (6.4, 12.0, 1.6), 8: (8.4, 16.0, 1.6),
}


def _screw(d: float, length: float, head: Sequence[Shape], bom: str) -> LibPart:
    if length <= 0:
        raise ValueError(f"screw length must be positive, got {length}")
    shape = assemble(*head, disk(d, length, -length))
    return LibPart(shape, bom, mass_g(shape, STEEL), {"head": z_frame(), "tip": z_frame(z=-length)})


def _hex_socket_head(dk: float, k: float, s: float, t: float) -> list[Solid]:
    """Cylindrical head split at the socket floor: solid base + ring with a blind hex socket."""
    return [disk(dk, k - t), disk(dk, t, k - t, holes=[regular_polygon(s)])]


def socket_head_screw(size: str | float, length: float) -> LibPart:
    """ISO 4762 socket head cap screw, e.g. `socket_head_screw("M3", 10)` (length under head)."""
    d = metric_size(size)
    dk, k, s, t = lookup_metric(_ISO4762, d, "socket_head_screw")
    head = _hex_socket_head(dk, k, s, t)
    return _screw(d, length, head, f"ISO 4762 M{d:g}x{length:g} socket head cap screw")


def button_head_screw(size: str | float, length: float) -> LibPart:
    """ISO 7380-1 button head screw (M3–M8); dome simplified to a cone frustum with a hex socket."""
    d = metric_size(size)
    dk, k, s, t = lookup_metric(_ISO7380, d, "button_head_screw")
    dome = Solid.make_cone(dk / 2, 0.6 * dk / 2, k) - prism(regular_polygon(s), t, k - t)
    return _screw(d, length, [dome], f"ISO 7380 M{d:g}x{length:g} button head screw")


def hex_nut(size: str | float) -> LibPart:
    """ISO 4032 hex nut, flats parallel to X, bearing face at z=0, bore = nominal diameter."""
    d = metric_size(size)
    s, m = lookup_metric(_ISO4032, d, "hex_nut")
    shape = assemble(prism(regular_polygon(s), m, bores=[(d, 0.0, 0.0)]))
    return LibPart(shape, f"ISO 4032 M{d:g} hex nut", mass_g(shape, STEEL),
                   {"bottom": z_frame(), "top": z_frame(z=m)})


def washer(size: str | float) -> LibPart:
    """ISO 7089 plain washer (normal series) lying on z=0."""
    d = metric_size(size)
    d1, d2, h = lookup_metric(_ISO7089, d, "washer")
    shape = assemble(tube(d2, d1, h))
    return LibPart(shape, f"ISO 7089 M{d:g} washer {d1:g}x{d2:g}x{h:g}", mass_g(shape, STEEL),
                   {"bottom": z_frame(), "top": z_frame(z=h)})
