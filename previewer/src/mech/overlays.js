// Scene overlays: clearance issue markers, probe paths, joint axes and the section plane.
import {
  ArrowHelper, BoxGeometry, BufferGeometry, Color, CylinderGeometry, DoubleSide, EdgesGeometry,
  Group, Line, LineBasicMaterial, LineSegments, Matrix4, Mesh, MeshBasicMaterial, PlaneGeometry,
  Quaternion, SphereGeometry, TorusGeometry, Vector3,
} from 'three';
import { STATUS_COLORS } from './parts.js';
import { fmt } from './model.js';

const PROBE_COLORS = [0x4dd0e1, 0xf06292, 0xffd54f, 0xa5d6a7, 0xb39ddb, 0xffab91];
const Y = new Vector3(0, 1, 0);

/** Marker material: always drawn on top so an issue is visible through the parts around it. */
const onTop = color => new MeshBasicMaterial({ color, depthTest: false, depthWrite: false, transparent: true, opacity: 0.95 });

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
   * @param {{a, b, status, pa, pb, distance, volume}[]} entries  scene.json issue entries
   * @param {object} opts  {offsetOf(id) → Vector3, size: mm, selected?: {entry?, box?: {center, size}, text}}
   */
  set(entries, { offsetOf, size, selected = null }) {
    this.clear();
    const r = size * 0.008;
    for (const e of entries) {
      if (e.status !== 'interference' && e.status !== 'tight') continue;
      const isSel = selected?.entry === e;
      const color = STATUS_COLORS[e.status];
      const pa = new Vector3(...e.pa).add(offsetOf(e.a));
      const pb = new Vector3(...e.pb).add(offsetOf(e.b));
      const k = isSel ? 1.6 : 1;
      for (const p of [pa, pb]) {
        const s = new Mesh(new SphereGeometry(r * k, 16, 12), onTop(color));
        s.position.copy(p);
        s.renderOrder = 10;
        this.group.add(s);
      }
      if (pa.distanceTo(pb) > 1e-6) this.group.add(this._segment(pa, pb, r * 0.35 * k, color));
      if (isSel) {
        const text = e.status === 'interference'
          ? `interference ${fmt(e.volume)} mm³ · ${e.a} / ${e.b}`
          : `tight ${fmt(e.distance)} mm · ${e.a} / ${e.b}`;
        this.label(pa.clone().add(pb).multiplyScalar(0.5), e.status, text);
      }
    }
    if (selected?.box) this._box(selected.box, selected.status || 'interference', r, selected.text, !selected.entry);
  }

  _segment(a, b, radius, color) {
    const len = a.distanceTo(b);
    const m = new Mesh(new CylinderGeometry(radius, radius, len, 10), onTop(color));
    m.position.copy(a).add(b).multiplyScalar(0.5);
    m.quaternion.copy(new Quaternion().setFromUnitVectors(Y, b.clone().sub(a).normalize()));
    m.renderOrder = 10;
    return m;
  }

  /** Overlap extent box (report issue `extent` at `location`); labelled when there is no pa/pb. */
  _box({ center, size, matrix }, status, r, text, withLabel) {
    const color = STATUS_COLORS[status] || STATUS_COLORS.interference;
    const s = size.clone().max(new Vector3(r, r, r));
    const geom = new BoxGeometry(s.x, s.y, s.z);
    const place = new Matrix4().copy(matrix).multiply(new Matrix4().makeTranslation(center.x, center.y, center.z));
    const fill = new Mesh(geom, new MeshBasicMaterial({ color, transparent: true, opacity: 0.22, depthTest: false, depthWrite: false }));
    const edges = new LineSegments(new EdgesGeometry(geom), new LineBasicMaterial({ color, depthTest: false, transparent: true }));
    for (const o of withLabel ? [fill, edges, new Mesh(new SphereGeometry(r * 1.4, 16, 12), onTop(color))] : [fill, edges]) {
      o.matrixAutoUpdate = false;
      o.matrix.copy(place);
      o.renderOrder = 10;
      this.group.add(o);
    }
    if (withLabel && text) this.label(center.clone().applyMatrix4(matrix), status, text);
  }
}

/** Probe trajectories over the whole study plus a dot at the current frame. */
export class ProbePaths extends Layer {
  set(probes, frame, { offsetOf, partOf, size, visible }) {
    this.clear();
    if (!visible) return;
    probes.forEach(({ name, pts }, i) => {
      const color = new Color(PROBE_COLORS[i % PROBE_COLORS.length]);
      const off = offsetOf(partOf(name));
      const points = pts.map(p => new Vector3(...p).add(off));
      const line = new Line(new BufferGeometry().setFromPoints(points),
        new LineBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false }));
      line.renderOrder = 8;
      this.group.add(line);
      const f = frame === null ? 0 : frame;
      const dot = new Mesh(new SphereGeometry(size * 0.008, 14, 10), onTop(color));
      dot.position.copy(points[Math.min(f, points.length - 1)]);
      dot.renderOrder = 9;
      this.group.add(dot);
      this.label(dot.position, 'probe', name).el.style.setProperty('--c', color.getStyle());
    });
  }
}

/** Joint axes at the current pose: revolute = arrow + ring, prismatic = double arrow. */
export class JointAxes extends Layer {
  set(joints, { transformOf, offsetOf, size, visible }) {
    this.clear();
    if (!visible) return;
    const len = size * 0.22;
    for (const j of joints) {
      if (j.kind === 'fixed') continue;
      const T = transformOf(j.parent);
      const origin = new Vector3(...j.origin).applyMatrix4(T).add(offsetOf(j.parent));
      const dir = new Vector3(...j.axis).transformDirection(T).normalize();
      const revolute = j.kind === 'revolute';
      const color = revolute ? 0xffcc33 : 0xc792ea;
      const start = origin.clone().addScaledVector(dir, revolute ? -len * 0.35 : 0);
      const arrow = new ArrowHelper(dir, start, revolute ? len * 1.35 : len, color, len * 0.18, len * 0.09);
      if (!revolute) {
        const back = new ArrowHelper(dir.clone().negate(), origin, len, color, len * 0.18, len * 0.09);
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
    this.group.traverse(o => {
      if (o.material) Object.assign(o.material, { depthTest: false, transparent: true });
      // ArrowHelper shares one line/cone geometry across instances: never dispose it.
      if (o.parent?.isArrowHelper || o.isArrowHelper) o.userData.sharedGeometry = true;
      o.renderOrder = 7;
    });
  }
}

/** Faint quad + outline where the section plane cuts the model bounds. */
export class SectionPlane extends Layer {
  set(section, bounds) {
    this.clear();
    if (!section) return;
    const size = bounds.getSize(new Vector3()).multiplyScalar(1.15);
    const c = bounds.getCenter(new Vector3());
    const dims = { x: [size.y, size.z], y: [size.x, size.z], z: [size.x, size.y] }[section.axis];
    const geom = new PlaneGeometry(dims[0], dims[1]);
    const plane = new Mesh(geom, new MeshBasicMaterial({
      color: 0x8ab4ff, transparent: true, opacity: 0.06, side: DoubleSide, depthWrite: false,
    }));
    const outline = new LineSegments(new EdgesGeometry(geom), new LineBasicMaterial({ color: 0x8ab4ff, transparent: true, opacity: 0.45 }));
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
