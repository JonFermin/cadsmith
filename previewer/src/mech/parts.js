// Part meshes: pose (T per frame + explode), styling (hide / isolate / issue colours),
// section clipping with back-face "caps", and translucent ghost poses.
import {
  BackSide, Box3, Color, EdgesGeometry, FrontSide, Group, LineBasicMaterial, LineSegments, Matrix4,
  Mesh, MeshBasicMaterial, MeshStandardMaterial, Vector3,
} from 'three';

export const STATUS_COLORS = { interference: new Color(0xff3b30), tight: new Color(0xff9f0a) };
const SECTION_TONE = new Color(0xf2ead8);
const EDGE_ANGLE = 28; // degrees; hides tessellation seams on curved faces

function partColor(hex) {
  try {
    return new Color(hex || '#9aa4b2');
  } catch {
    return new Color('#9aa4b2');
  }
}

/** Flat, lighter tone for section faces so cuts read differently from shaded surfaces. */
function capColor(color) {
  return color.clone().lerp(SECTION_TONE, 0.45);
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
        color, metalness: 0.08, roughness: 0.58, side: FrontSide,
        transparent: opacity < 1, opacity,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, // edges sit on top
      });
      const mesh = new Mesh(geom, material);
      mesh.matrixAutoUpdate = false;
      mesh.userData.partId = meta.id;
      const edgeGeom = new EdgesGeometry(geom, EDGE_ANGLE);
      const edgeMat = new LineBasicMaterial({ color: color.clone().multiplyScalar(0.35), transparent: true, opacity: 0.7 });
      const edges = new LineSegments(edgeGeom, edgeMat);
      edges.raycast = () => {};
      // Back faces seen through a section cut read as a solid, darker "cap".
      const capMat = new MeshBasicMaterial({ color: capColor(color), side: BackSide });
      const cap = new Mesh(geom, capMat);
      cap.visible = false;
      cap.raycast = () => {};
      mesh.add(edges, cap);
      this.group.add(mesh);
      const homeCenter = geom.boundingBox.getCenter(new Vector3());
      this.items.set(meta.id, { meta, geom, edgeGeom, mesh, edges, cap, material, edgeMat, capMat, color, opacity, homeCenter,
        offset: new Vector3(), T: new Matrix4() });
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
      it.edgeMat.color.copy(color).multiplyScalar(0.35);
      it.edgeMat.opacity = context ? 0.12 : 0.7;
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

  /** World AABB of the given parts (default: all visible) at the current pose. */
  currentBox(ids = null) {
    const box = new Box3();
    for (const [id, it] of this.items) {
      if (ids ? !ids.includes(id) : !it.mesh.visible) continue;
      box.union(it.geom.boundingBox.clone().applyMatrix4(it.mesh.matrix));
    }
    return box;
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
          color: it.color.clone().lerp(new Color(0xffffff), 0.25), transparent: true,
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
