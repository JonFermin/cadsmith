// Part meshes: pose (T per frame + explode), styling (hide / isolate / issue colours),
// section clipping with back-face "caps", and translucent ghost poses.
import {
  BackSide, Box3, Color, EdgesGeometry, FrontSide, Group, LineBasicMaterial, LineSegments, Matrix4,
  Mesh, MeshBasicMaterial, MeshStandardMaterial, Vector3,
} from 'three';
import { srgb } from './viewer.js';

// All Colors here are linear (the renderer outputs sRGB); hex constants go through srgb().
export const STATUS_COLORS = { interference: srgb(0xff3b30), tight: srgb(0xff9f0a) };
const SECTION_TONE = srgb(0xf2ead8);
const WHITE = new Color(1, 1, 1);
const EDGE_ANGLE = 28; // degrees; hides tessellation seams on curved faces
const DEFAULT_COLOR = '#9aa4b2';

function partColor(hex) {
  try {
    return srgb(hex || DEFAULT_COLOR);
  } catch {
    return srgb(DEFAULT_COLOR);
  }
}

/** Flat, lighter tone for section faces so cuts read differently from shaded surfaces. */
function capColor(color) {
  return color.clone().lerp(SECTION_TONE, 0.45);
}

// 13 axes (cube faces, edges and corners). A part's extreme vertices along ±each are ≤ 26 points
// whose hull hugs the part (a 26-DOP) — tight even for a rod drawn at an angle, whose home-pose
// AABB (all an STL bbox gives) is loose. Camera fits, section sizing and ghost framing use them.
const DOP_AXES = [
  [1, 0, 0], [0, 1, 0], [0, 0, 1],
  [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
  [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
];

/** The extreme vertices of a geometry along ±DOP_AXES (home frame), deduplicated. */
export function extremePoints(geom) {
  const a = geom.attributes.position;
  const n = a.count;
  if (!n) return [];
  const lo = new Float64Array(13).fill(Infinity);
  const hi = new Float64Array(13).fill(-Infinity);
  const loI = new Int32Array(13);
  const hiI = new Int32Array(13);
  for (let i = 0; i < n; i++) {
    const x = a.getX(i);
    const y = a.getY(i);
    const z = a.getZ(i);
    for (let k = 0; k < 13; k++) {
      const [u, v, w] = DOP_AXES[k];
      const d = u * x + v * y + w * z;
      if (d < lo[k]) { lo[k] = d; loI[k] = i; }
      if (d > hi[k]) { hi[k] = d; hiI[k] = i; }
    }
  }
  return [...new Set([...loI, ...hiI])].map(i => new Vector3(a.getX(i), a.getY(i), a.getZ(i)));
}

/**
 * Where `plane` crosses the segments joining points on opposite sides of it. Taken over every
 * pair, the hull of the result is exactly the section of hull(pts) by the plane.
 */
export function planeCrossings(pts, plane, out = []) {
  const d = pts.map(p => plane.distanceToPoint(p));
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      if ((d[i] >= 0) !== (d[j] >= 0)) out.push(pts[i].clone().lerp(pts[j], d[i] / (d[i] - d[j])));
    }
  }
  return out;
}

/**
 * What is left of hull(pts) on the kept side of a clipping `plane` (distance ≥ 0), as points
 * whose hull is exactly hull(pts) ∩ half-space: the kept points plus the plane crossings.
 */
export function clipPoints(pts, plane, out = []) {
  for (const p of pts) if (plane.distanceToPoint(p) >= 0) out.push(p);
  return planeCrossings(pts, plane, out);
}

/** Edge lines: darker than the face on light parts, lighter on dark parts (so black parts still read). */
function edgeColor(color) {
  const lum = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
  return lum < 0.05 ? color.clone().lerp(WHITE, 0.32) : color.clone().multiplyScalar(0.3);
}

export class PartSet {
  /**
   * @param {import('./model.js').MechModel} model
   * @param {Map<string, import('three').BufferGeometry>} geometries
   */
  constructor(model, geometries) {
    this.model = model;
    this.group = new Group();
    this.ghostGroup = new Group();
    this.items = new Map();
    this.clipping = [];
    for (const meta of model.parts) {
      const geom = geometries.get(meta.id);
      const color = partColor(meta.color);
      const opacity = meta.opacity ?? 1;
      const material = new MeshStandardMaterial({
        color, metalness: 0.06, roughness: 0.55, side: FrontSide,
        transparent: opacity < 1, opacity, clipShadows: true,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, // edges sit on top
      });
      const mesh = new Mesh(geom, material);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData.partId = meta.id;
      const edgeGeom = new EdgesGeometry(geom, EDGE_ANGLE);
      const edgeMat = new LineBasicMaterial({ color: edgeColor(color), transparent: true, opacity: 0.8 });
      const edges = new LineSegments(edgeGeom, edgeMat);
      edges.raycast = () => {};
      // Back faces seen through a section cut read as a solid, lighter "cap".
      const capMat = new MeshBasicMaterial({ color: capColor(color), side: BackSide });
      const cap = new Mesh(geom, capMat);
      cap.visible = false;
      cap.raycast = () => {};
      mesh.add(edges, cap);
      this.group.add(mesh);
      const homeCenter = geom.boundingBox.getCenter(new Vector3());
      const hull = extremePoints(geom);
      this.items.set(meta.id, { meta, geom, edgeGeom, mesh, edges, cap, material, edgeMat, capMat, color, opacity, homeCenter,
        hull, offset: new Vector3(), T: new Matrix4() });
    }
  }

  /**
   * Place every part at (study, frame) and push parts apart by `explode` × their offset from
   * the assembly centre (computed at the current pose, so it follows the motion).
   */
  setPose(si, frame, explode = 0) {
    const centers = new Map();
    const all = new Box3();
    for (const [id, it] of this.items) {
      it.T.copy(this.model.transform(si, frame, id));
      const c = it.homeCenter.clone().applyMatrix4(it.T);
      centers.set(id, c);
      all.expandByPoint(c);
    }
    const mid = all.getCenter(new Vector3());
    for (const [id, it] of this.items) {
      it.offset.copy(centers.get(id)).sub(mid).multiplyScalar(explode);
      it.mesh.matrix.makeTranslation(it.offset.x, it.offset.y, it.offset.z).multiply(it.T);
      it.mesh.matrixWorldNeedsUpdate = true;
    }
  }

  offsetOf(id) {
    return this.items.get(id)?.offset || new Vector3();
  }

  /**
   * @param {{hidden: Set<string>, isolate: Set<string>, highlight: Map<string, string>,
   *          xray: Set<string>, section: boolean}} s
   */
  applyStyle({ hidden, isolate, highlight, xray, section }) {
    for (const [id, it] of this.items) {
      const context = isolate.size > 0 && !isolate.has(id); // faint, for orientation only
      it.mesh.visible = !hidden.has(id);
      const status = highlight.get(id);
      const color = it.color.clone();
      if (status) color.lerp(STATUS_COLORS[status], 0.8);
      it.material.color.copy(color);
      it.material.emissive.copy(status ? STATUS_COLORS[status] : new Color(0)).multiplyScalar(status ? 0.22 : 0);
      const opacity = context ? Math.min(it.opacity, 0.1) : xray.has(id) ? Math.min(it.opacity, 0.45) : it.opacity;
      it.material.opacity = opacity;
      it.material.transparent = opacity < 1;
      it.material.depthWrite = opacity >= 1;
      it.mesh.castShadow = !context && opacity > 0.3;
      it.edgeMat.color.copy(edgeColor(color));
      it.edgeMat.opacity = context ? 0.12 : 0.8;
      it.cap.visible = section && !context;
      it.capMat.color.copy(capColor(color));
      it.material.clippingPlanes = this.clipping;
      it.edgeMat.clippingPlanes = this.clipping;
      it.capMat.clippingPlanes = this.clipping;
      it.mesh.renderOrder = context ? 2 : 0;
    }
  }

  setClipping(planes) {
    this.clipping = planes;
  }

  /** Parts to consider: the given ids, else every visible one. */
  _select(ids) {
    const out = [];
    for (const [id, it] of this.items) {
      if (ids ? ids.includes(id) : it.mesh.visible) out.push(it);
    }
    return out;
  }

  /** A part's hull points (extremePoints) placed by `matrix` (default: its current pose). */
  _hull(it, matrix = it.mesh.matrix) {
    return it.hull.map(p => p.clone().applyMatrix4(matrix));
  }

  /** World AABB of the given parts (default: all visible) at the current pose. */
  currentBox(ids = null) {
    const box = new Box3();
    for (const it of this._select(ids)) {
      for (const p of this._hull(it)) box.expandByPoint(p);
    }
    return box;
  }

  /**
   * Hull points of each part at the current (possibly exploded) pose — the camera fit target.
   * With a section `plane` only what is left on the kept side counts (a part cut away entirely
   * adds nothing).
   */
  hullPoints(ids = null, out = [], plane = null) {
    for (const it of this._select(ids)) {
      const pts = this._hull(it);
      if (plane) clipPoints(pts, plane, out);
      else out.push(...pts);
    }
    return out;
  }

  /**
   * Points where a section `plane` crosses the given parts (default: all visible) at the current
   * pose — the extent of the cut, which the section overlay is sized to.
   */
  cutPoints(ids, plane, out = []) {
    for (const it of this._select(ids)) planeCrossings(this._hull(it), plane, out);
    return out;
  }

  /** Hull points of the moving parts at the given frames (what ghost poses occupy). */
  ghostPoints(si, frames, hidden, out = [], plane = null) {
    if (si === null) return out;
    for (const id of this.model.movingParts(si)) {
      if (hidden.has(id)) continue;
      const it = this.items.get(id);
      for (const f of frames) {
        const pts = this._hull(it, this.model.transform(si, f, id));
        if (plane) clipPoints(pts, plane, out);
        else out.push(...pts);
      }
    }
    return out;
  }

  /** AABB of the parts over every frame of a study (the motion envelope), no explode. */
  envelope(si, ids = null) {
    const box = new Box3();
    const frames = Math.max(this.model.frameCount(si), 1);
    for (const [id, it] of this.items) {
      if (ids ? !ids.includes(id) : !it.mesh.visible) continue;
      for (let f = 0; f < frames; f++) {
        const T = this.model.transform(si, si === null ? null : f, id);
        box.union(it.geom.boundingBox.clone().applyMatrix4(T));
      }
    }
    return box;
  }

  /** Translucent copies of the moving parts at `frames`; opacity ramps up with time. */
  setGhosts(si, frames, hidden) {
    for (const child of this.ghostGroup.children) {
      child.material.dispose();
      for (const c of child.children) c.material.dispose();
    }
    this.ghostGroup.clear();
    const moving = this.model.movingParts(si).filter(id => !hidden.has(id));
    frames.forEach((f, i) => {
      const k = frames.length > 1 ? i / (frames.length - 1) : 1;
      for (const id of moving) {
        const it = this.items.get(id);
        const mat = new MeshStandardMaterial({
          color: it.color, metalness: 0, roughness: 0.7, transparent: true,
          opacity: 0.07 + 0.13 * k, depthWrite: false, clippingPlanes: this.clipping,
        });
        const ghost = new Mesh(it.geom, mat);
        ghost.matrixAutoUpdate = false;
        ghost.matrix.copy(this.model.transform(si, f, id));
        ghost.raycast = () => {};
        const edgeMat = new LineBasicMaterial({
          color: it.color.clone().lerp(WHITE, 0.25), transparent: true,
          opacity: 0.16 + 0.22 * k, depthWrite: false, clippingPlanes: this.clipping,
        });
        ghost.add(new LineSegments(it.edgeGeom, edgeMat));
        ghost.renderOrder = 1;
        this.ghostGroup.add(ghost);
      }
    });
  }

  meshes() {
    return [...this.items.values()].map(it => it.mesh);
  }
}
