"""Standard-parts library: real standard dimensions, simplified geometry (no threads).

Every constructor returns a `LibPart` (shape + BOM line + mass + named frames) that can be placed
with `Pos(...) * Rot(...) * part` or `part.mate("frame", target_location)` and handed straight to
`Assembly.part(name, libpart)`. Never uses bd_warehouse.
"""
from .bearings import bearing
from .fasteners import button_head_screw, hex_nut, socket_head_screw, washer
from .fits import (clearance_hole, fdm_hole, fit, heat_set_insert, it_grade, tap_drill,
                   tolerance_zone)
from .gears import GearPair, gear_pair, gt2_pulley, rack, spur_gear
from .libpart import LibPart
from .motion_parts import T8_LEAD, extrusion_2020, extrusion_2040, rod, t8_leadscrew, t8_nut
from .motors import mg996r, n20_gearmotor, nema17, nema23, sg90

__all__ = [
    "LibPart", "GearPair",
    "fit", "tolerance_zone", "it_grade", "fdm_hole", "clearance_hole", "tap_drill",
    "heat_set_insert",
    "socket_head_screw", "button_head_screw", "hex_nut", "washer",
    "bearing",
    "nema17", "nema23", "sg90", "mg996r", "n20_gearmotor",
    "spur_gear", "gear_pair", "rack", "gt2_pulley",
    "rod", "t8_leadscrew", "t8_nut", "extrusion_2020", "extrusion_2040", "T8_LEAD",
]
