// Scene overlays: clearance issue markers, probe paths, joint axes and the section plane.
import {
  ArrowHelper, BoxGeometry, BufferGeometry, Color, CylinderGeometry, DoubleSide, EdgesGeometry,
  Group, Line, LineBasicMaterial, LineSegments, Matrix4, Mesh, MeshBasicMaterial, PlaneGeometry,
  Quaternion, SphereGeometry, TorusGeometry, Vector3, Vector4,
} from 'three';
import { STATUS_COLORS } from './parts.js';
import { fmt } from './model.js';
import { srgb } from './viewer.js';

const PROBE_COLORS = [0x4dd0e1, 0xf06292, 0xffd54f, 0xa5d6a7, 0xb39ddb, 0xffab91];
const Y = new Vector3(0, 1, 0);

/** Marker material (`color` linear): drawn on top so an issue is visible through the parts around it. */
const onTop = (color, opacity = 0.95) => new MeshBasicMaterial({ color, depthTest: false, depthWrite: false, transparent: true, opacity });

// Issue and probe markers keep a fixed size on screen (CSS px) in every pane, whatever the scene
// size or zoom, and are see-through: a marker must point at an overlap, never hide it.
const MARKER_PX = 5;
const MARKER_OPACITY = 0.5;
const _vp = new Vector4();
const _pos = new Vector3();

/** World length of one CSS pixel at `pos` for `camera`, in the renderer's current viewport. */
function worldPerPixel(renderer, camera, pos) {
  renderer.getViewport(_vp);
  const h = Math.max(_vp.w, 1);
  if (camera.isOrthographicCamera) return (camera.top - camera.bottom) / camera.zoom / h;
  const depth = Math.abs(_pos.copy(pos).applyMatrix4(camera.matrixWorldInverse).z);
  return (2 * depth * Math.tan((camera.fov * Math.PI) / 360)) / h;
}

/**
 * Rescale `obj` (unit-sized geometry) to `px` screen pixels right before each camera draws it;
 * `radial` scales only X/Z (a cylinder along its Y axis keeps its length).
 */
function screenSized(obj, px, radial = false) {
  obj.frustumCulled = false; // its bounding sphere follows the last camera's scale
  obj.onBeforeRender = (renderer, _scene, camera) => {
    const k = px * worldPerPixel(renderer, camera, obj.getWorldPosition(new Vector3()));
    if (radial) obj.scale.set(k, 1, k);
    else obj.scale.setScalar(k);
    obj.updateMatrixWorld();
  };
  return obj;
}

/** A see-through dot `px` pixels in radius at `pos`. */
function dot(pos, color, px, opacity = MARKER_OPACITY) {
  const m = new Mesh(new SphereGeometry(1, 16, 12), onTop(color, opacity));
  m.position.copy(pos);
  return screenSized(m, px);
}

function el(cls, text) {
  const e = document.createElement('div');
  e.className = cls;
  e.textContent = text;
  return e;
}

function disposeTree(obj) {
  obj.traverse(o => {
    if (o.geometry && !o.userData.sharedGeometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
}

class Layer {
  constructor(viewer) {
    this.viewer = viewer;
    this.group = new Group();
    this.labels = [];
    viewer.content.add(this.group);
  }

  clear() {
    disposeTree(this.group);
    this.group.clear();
    for (const l of this.labels) this.viewer.removeLabel(l);
    this.labels = [];
  }

  label(pos, cls, text) {
    const l = this.viewer.addLabel(el(`label ${cls}`, text));
    l.pos.copy(pos);
    this.labels.push(l);
    return l;
  }
}

/** Clearance problems at the current pose: segment + spheres pa→pb, optional overlap box. */
export class IssueMarkers extends Layer {
  /**
   * Markers are drawn at a fixed screen size (MARKER_PX, 1.5× when selected) and see-through, so
   * they point at a small overlap without covering it; the selected issue's overlap box is drawn
   * at its true extent.
   * @param {{a, b, status, pa, pb, distance, volume}[]} entries  scene.json issue entries
   * @param {object} opts  {offsetOf(id) → Vector3, size: mm, selected?: {entry?, box?: {center, size}, text}}
   */
  set(entries, { offsetOf, size, selected = null }) {
    this.clear();
    const r = size * 0.002; // only the smallest overlap box drawn (a zero extent stays visible)
    for (const e of entries) {
      if (e.status !== 'interference' && e.status !== 'tight') continue;
      const isSel = selected?.entry === e;
      const color = STATUS_COLORS[e.status];
      const pa = new Vector3(...e.pa).add(offsetOf(e.a));
      const pb = new Vector3(...e.pb).add(offsetOf(e.b));
      const px = MARKER_PX * (isSel ? 1.5 : 1);
      for (const p of [pa, pb]) {
        const s = dot(p, color, px, isSel ? 0.65 : MARKER_OPACITY);
        s.renderOrder = 10;
        this.group.add(s);
      }
      if (pa.distanceTo(pb) > 1e-6) this.group.add(this._segment(pa, pb, px * 0.35, color));
      if (isSel) {
        const text = e.status === 'interference'
          ? `interference ${fmt(e.volume)} mm³ · ${e.a} / ${e.b}`
          : `tight ${fmt(e.distance)} mm · ${e.a} / ${e.b}`;
        this.label(pa.clone().add(pb).multiplyScalar(0.5), e.status, text);
      }
    }
    if (selected?.box) this._box(selected.box, selected.status || 'interference', r, selected.text, !selected.entry);
  }

  _segment(a, b, px, color) {
    const len = a.distanceTo(b);
    const m = new Mesh(new CylinderGeometry(1, 1, len, 10), onTop(color, 0.8));
    m.position.copy(a).add(b).multiplyScalar(0.5);
    m.quaternion.copy(new Quaternion().setFromUnitVectors(Y, b.clone().sub(a).normalize()));
    m.renderOrder = 10;
    return screenSized(m, px, true);
  }

  /** Overlap extent box (report issue `extent` at `location`); labelled when there is no pa/pb. */
  _box({ center, size, matrix }, status, r, text, withLabel) {
    const color = STATUS_COLORS[status] || STATUS_COLORS.interference;
    const s = size.clone().max(new Vector3(r, r, r));
    const geom = new BoxGeometry(s.x, s.y, s.z);
    const place = new Matrix4().copy(matrix).multiply(new Matrix4().makeTranslation(center.x, center.y, center.z));
    const fill = new Mesh(geom, new MeshBasicMaterial({ color, transparent: true, opacity: 0.22, depthTest: false, depthWrite: false }));
    const edges = new LineSegments(new EdgesGeometry(geom), new LineBasicMaterial({ color, depthTest: false, transparent: true }));
    for (const o of [fill, edges]) {
      o.matrixAutoUpdate = false;
      o.matrix.copy(place);
      o.renderOrder = 10;
      this.group.add(o);
    }
    const at = center.clone().applyMatrix4(matrix);
    if (withLabel) {
      const d = dot(at, color, MARKER_PX * 1.5, 0.65);
      d.renderOrder = 10;
      this.group.add(d);
      if (text) this.label(at, status, text);
    }
  }
}

/**
 * Probe trajectories over the whole study plus a dot at the current frame. A section's `clip`
 * planes cut them like the parts, so a section view shows no paths of parts it has removed.
 */
export class ProbePaths extends Layer {
  set(probes, frame, { offsetOf, partOf, size, visible, clip = [] }) {
    this.clear();
    if (!visible) return;
    probes.forEach(({ name, pts }, i) => {
      const hex = PROBE_COLORS[i % PROBE_COLORS.length];
      const color = srgb(hex);
      const off = offsetOf(partOf(name));
      const points = pts.map(p => new Vector3(...p).add(off));
      const line = new Line(new BufferGeometry().setFromPoints(points),
        new LineBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false, clippingPlanes: clip }));
      line.renderOrder = 8;
      this.group.add(line);
      const f = frame === null ? 0 : frame;
      const at = dot(points[Math.min(f, points.length - 1)], color, MARKER_PX, 0.9);
      at.material.clippingPlanes = clip;
      at.renderOrder = 9;
      this.group.add(at);
      if (clip.every(pl => pl.distanceToPoint(at.position) >= 0)) {
        this.label(at.position, 'probe', name).el.style.setProperty('--c', new Color(hex).getStyle());
      }
    });
  }
}

/**
 * Joint axes at the current pose: revolute = arrow + ring, prismatic = double arrow. A ball()
 * joint is three revolutes chained through virtual knuckle bodies that the scene does not carry
 * (no mesh, no transforms), so their axes would sit at the home pose: those joints are skipped and
 * the ball is drawn instead — three rings around its center, which moves with the ball's parent.
 */
export class JointAxes extends Layer {
  set(joints, { transformOf, offsetOf, size, visible, balls = [], isPart = () => true }) {
    this.clear();
    if (!visible) return;
    const len = size * 0.22;
    for (const j of joints) {
      if (j.kind === 'fixed') continue;
      if (!isPart(j.parent) || !isPart(j.child)) continue; // a ball()'s revolute: drawn as its ball below
      const T = transformOf(j.parent);
      const origin = new Vector3(...j.origin).applyMatrix4(T).add(offsetOf(j.parent));
      const dir = new Vector3(...j.axis).transformDirection(T).normalize();
      const revolute = j.kind === 'revolute';
      const color = srgb(revolute ? 0xffcc33 : 0xc792ea);
      const start = origin.clone().addScaledVector(dir, revolute ? -len * 0.35 : 0);
      const arrow = new ArrowHelper(dir, start, revolute ? len * 1.35 : len, 0xffffff, len * 0.18, len * 0.09);
      arrow.setColor(color);
      if (!revolute) {
        const back = new ArrowHelper(dir.clone().negate(), origin, len, 0xffffff, len * 0.18, len * 0.09);
        back.setColor(color);
        this.group.add(back);
      }
      this.group.add(arrow);
      if (revolute) {
        const ring = new Mesh(new TorusGeometry(len * 0.16, len * 0.012, 8, 40), new MeshBasicMaterial({ color }));
        ring.position.copy(origin);
        ring.quaternion.setFromUnitVectors(new Vector3(0, 0, 1), dir);
        this.group.add(ring);
      }
      this.label(origin.clone().addScaledVector(dir, revolute ? len : len * 1.05), 'joint', j.name);
    }
    for (const b of balls) {
      if (!isPart(b.parent)) continue;
      const T = transformOf(b.parent);
      const center = new Vector3(...b.center).applyMatrix4(T).add(offsetOf(b.parent));
      const color = srgb(0xffcc33);
      const rot = new Quaternion().setFromRotationMatrix(new Matrix4().extractRotation(T));
      for (const n of [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)]) {
        const ring = new Mesh(new TorusGeometry(len * 0.16, len * 0.012, 8, 40), new MeshBasicMaterial({ color }));
        ring.position.copy(center);
        ring.quaternion.setFromUnitVectors(new Vector3(0, 0, 1), n.applyQuaternion(rot));
        this.group.add(ring);
      }
      this.label(center.clone().add(new Vector3(0, 0, len * 0.3)), 'joint', b.name);
    }
    this.group.traverse(o => {
      if (o.material) Object.assign(o.material, { depthTest: false, transparent: true });
      // ArrowHelper shares one line/cone geometry across instances: never dispose it.
      if (o.parent?.isArrowHelper || o.isArrowHelper) o.userData.sharedGeometry = true;
      o.renderOrder = 7;
    });
  }
}

/**
 * Faint quad + thin outline where the section plane cuts the parts: `box` is the extent of the
 * cut at the current pose (MechApp._sectionBox), never the motion envelope, so the quad hugs the
 * cut faces instead of dwarfing the model; the caps themselves carry the cut.
 */
export class SectionPlane extends Layer {
  set(section, box) {
    this.clear();
    if (!section || box.isEmpty()) return;
    const size = box.getSize(new Vector3());
    const pad = size.length() * 0.025;
    size.multiplyScalar(1.02).addScalar(pad);
    const c = box.getCenter(new Vector3());
    const dims = { x: [size.y, size.z], y: [size.x, size.z], z: [size.x, size.y] }[section.axis];
    const geom = new PlaneGeometry(dims[0], dims[1]);
    const tint = srgb(0x8ab4ff);
    const plane = new Mesh(geom, new MeshBasicMaterial({
      color: tint, transparent: true, opacity: 0.03, side: DoubleSide, depthWrite: false,
    }));
    const outline = new LineSegments(new EdgesGeometry(geom), new LineBasicMaterial({ color: tint, transparent: true, opacity: 0.22 }));
    const g = new Group();
    g.add(plane, outline);
    g.position.copy(c);
    g.position[section.axis] = section.offset;
    const normal = new Vector3();
    normal[section.axis] = 1;
    g.quaternion.setFromUnitVectors(new Vector3(0, 0, 1), normal); // PlaneGeometry faces +Z
    g.renderOrder = 5;
    this.group.add(g);
  }
}
