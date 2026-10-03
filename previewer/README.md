# cadsmith Previewer

Interactive Three.js 3D previewer for cadsmith scene manifests.

## Setup

```bash
cd previewer
npm install
npm run dev
```

Opens at http://localhost:3000

## Loading a manifest

Three ways to load:

1. **File picker** — Click the file input in the toolbar
2. **Drag & drop** — Drop a `_manifest.json` file onto the viewport
3. **URL param** — `http://localhost:3000?manifest=../output/my_part_manifest.json`

## Controls

- **Drag** — Rotate camera
- **Shift+Drag** — Pan
- **Scroll** — Zoom
- **Touch** — Single finger rotate, pinch to zoom

## Manifest format

```json
{
  "name": "My Part",
  "units": "mm",
  "parts": [
    {
      "id": "base",
      "label": "Base Plate",
      "type": "box",
      "params": { "width": 50, "height": 5, "depth": 30 },
      "position": [0, 2.5, 0],
      "rotation": [0, 0, 0],
      "color": "#4488cc",
      "opacity": 1.0
    }
  ],
  "camera": {
    "distance": 120,
    "angle": [30, 45]
  }
}
```

### Geometry types

| type | params |
|------|--------|
| `box` | `width`, `height`, `depth` |
| `sphere` | `radius`, `segments` (optional) |
| `cylinder` | `radiusTop`, `radiusBottom`, `height`, `segments` (optional) |
| `extrude` | `shape` (array of [x,y] points), `height` |
| `lathe` | `points` (array of [x,y] profile points), `segments` |

### Difference operations

Parts with `"operation": "difference"` render as transparent red wireframes to indicate subtracted volumes:

```json
{
  "id": "hole",
  "label": "Mounting Hole",
  "type": "cylinder",
  "params": { "radiusTop": 2, "radiusBottom": 2, "height": 7 },
  "position": [10, 2.5, 0],
  "color": "#ff4444",
  "opacity": 0.3,
  "wireframe": true,
  "operation": "difference"
}
```

### Coordinate system

The previewer uses Three.js Y-up coordinates. When cadsmith generates manifests from build123d (Z-up), it swaps Y/Z automatically.

## Mechanism viewer (`mech.html`)

`/mech.html?m=<name>` shows `output/<name>.mech/` (written by `uv run mech run`; Z-up). It also
works from the static build (`npm run build` → `dist/`) with `output/` served at `/output/`, which
is what `uv run mech shot` does. Every parameter is checked: an unknown name or a malformed value
is an error (with a did-you-mean) in `window.__mechError`, never a silently different picture;
`window.__mechReady` turns true once the requested state has rendered.

| param | meaning |
|-------|---------|
| `m` | mechanism slug (`output/<m>.mech/`) |
| `study`, `frame` | study name; frame index of that study, or `home` for the drawn pose |
| `q=j:45,k:10` | the frame whose joint values are nearest these (revolute joints compare mod 360°); not with `frame=` (both pick the frame: an error) |
| `issue=i` | report issue *i*: its study + frame, its parts isolated and framed, the overlap marked |
| `hide`, `isolate` | part lists; isolated parts stay solid, the rest turn faint context |
| `focus=a,b` | frame these parts (in every pane of the quad layout) instead of the whole model |
| `view` | `iso` (default) `top` `front` `right` `left` `back` `bottom` — world-axis views: `front` looks along +Y |
| `cam=az,el` | camera direction in degrees (azimuth from +X toward +Y, elevation above XY); overrides `view` |
| `zoom=<factor>` | on the default fit: `2` = twice as close, `0.5` = twice as far (all panes) |
| `section=x[:mm]` | section plane ⊥ X/Y/Z (`-x` keeps the other side); without an offset it cuts through the parts of interest (an issue's overlap, else the range all focus/isolate parts share, else the visible parts' centre) |
| `explode=0.5` | push parts apart from the assembly centre |
| `ghost=N` | N evenly spaced poses of the study as translucent overlays (and framed) |
| `paths=0\|1` | probe trajectories: absent = drawn, framed only when they stay near the parts; `1` = always framed; `0` = hidden |
| `axes=1` | joint axis arrows (a `ball()` joint: three rings at its centre, which moves with its parent) |
| `layout=quad` | iso / top / front / right panes on one canvas |
| `ui=0` | canvas only (no panels / HUD) |

Framing: the camera fits the parts visible at the current pose (what is left of them after a
section cut) to ~72 % of the pane in its limiting direction — not the motion envelope; ghost poses count only with `ghost=N`,
probe paths as above, and `focus=` / `issue=` / `isolate=` frame just those parts. The grid is
sized to the model's home pose.
