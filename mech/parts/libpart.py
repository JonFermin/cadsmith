"""`LibPart`: a standard part's shape bundled with its BOM line, mass and named frames."""
from __future__ import annotations

from dataclasses import dataclass, field

from build123d import Location, Shape


@dataclass
class LibPart:
    """A library part modeled in place, plus named attachment frames.

    `frames` are `Location`s in the same (world) coordinates as `shape`; each frame's Z axis is the
    natural axis of that feature (shaft axis, screw axis pointing out of the hole, ...), so a frame
    can be handed straight to `Assembly.revolute(..., at=part.frames["shaft"])`.

    `mass_g` is set for bought-in metal parts (catalog or density-derived); `None` means "derive
    from the assembly part's material", the right default for parts that are usually printed.

    `kind` tags tooth-bearing parts ("gear", "rack", "pulley"): `Assembly.meshing_pairs` excuses
    only the tagged parts of two gear/rack-coupled bodies, so a crank fixed to a pinion is still
    clearance-checked against the mating gear.
    """

    shape: Shape
    bom: str
    mass_g: float | None = None
    frames: dict[str, Location] = field(default_factory=dict)
    kind: str | None = None

    def moved(self, loc: Location) -> LibPart:
        """Copy rigidly moved by `loc` (applied in world coordinates to shape and frames)."""
        return LibPart(
            shape=self.shape.moved(loc),
            bom=self.bom,
            mass_g=self.mass_g,
            frames={name: loc * frame for name, frame in self.frames.items()},
            kind=self.kind,
        )

    def __rmul__(self, loc: Location) -> LibPart:
        # `Pos(...) * Rot(...) * nema17()`: Location.__mul__ defers foreign types to us.
        if not isinstance(loc, Location):
            return NotImplemented
        return self.moved(loc)

    def mate(self, frame: str, target: Location) -> LibPart:
        """Move the part so that `frames[frame]` lands on `target` (position and orientation)."""
        try:
            current = self.frames[frame]
        except KeyError:
            raise KeyError(f"unknown frame {frame!r}; frames: {sorted(self.frames)}") from None
        return self.moved(target * current.inverse())
