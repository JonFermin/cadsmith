// Renderer, cameras, lights, grid, ground shadow and viewport layout (single or 2×2 quad). Z is up.
import {
  AmbientLight, ArrowHelper, BackSide, Box3, Color, CubeCamera, DirectionalLight, DoubleSide,
  Float32BufferAttribute, GridHelper, Group, HemisphereLight, LinearMipmapLinearFilter, Mesh, MeshBasicMaterial,
  OrthographicCamera, PCFSoftShadowMap, PerspectiveCamera, PlaneGeometry, Scene, ShadowMaterial, Sphere,
  SphereGeometry, sRGBEncoding, Vector3, WebGLCubeRenderTarget, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const BG = 0x15171c;
const DIVIDER = 0x08090b;
const Z_UP = new Vector3(0, 0, 1);
// Fraction of the pane the fitted points span in their limiting direction (the rest is margin).
const FILL_PERSPECTIVE = 0.72;
const FILL_ORTHO = 0.78;
const MAX_DRAWN_PX = 8.3e6;
// Key light from the front-right, falling more on the −Y face than the +X face so the two side
// faces an iso view shows differ in tone (balance in Viewer._addLights).
const KEY_DIR = new Vector3(0.4, -0.85, 1.1).normalize();
// The shadows come from a separate, steeper light from the same side: a short, compact shadow
// under each part grounds it (and darkens contacts) instead of a long cast shadow off the model.
const SHADOW_DIR = new Vector3(0.17, -0.34, 0.92).normalize();

/**
 * Colour for a material from a CSS hex / number. The renderer outputs sRGB, so every material
 * colour must be linear for the hex to come out as written on screen.
 */
export const srgb = c => new Color(c).convertSRGBToLinear();

/** Unit vector from the target toward the camera for each named view. */
const VIEW_DIRS = {
  iso: new Vector3(1, -1, 0.85).normalize(),
  top: new Vector3(0, -1e-4, 1).normalize(), // tiny −Y tilt keeps +Y up on screen
  bottom: new Vector3(0, -1e-4, -1).normalize(),
  front: new Vector3(0, -1, 0),
  back: new Vector3(0, 1, 0),
  right: new Vector3(1, 0, 0),
  left: new Vector3(-1, 0, 0),
};

export function camDirection(az, el) {
  const a = (az * Math.PI) / 180;
  const e = (Math.max(-89.9, Math.min(89.9, el)) * Math.PI) / 180;
  return new Vector3(Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e));
}

export function viewDirection(name) {
  return VIEW_DIRS[name].clone();
}

/** A "nice" step (1, 2 or 5 × 10^k) near `x`. */
export function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x));
  const m = x / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

/**
 * A tiny studio environment for specular reflections: a dome bright overhead, dim at the horizon
 * and dark below, plus soft boxes on the key and rim sides. Rendered once into a 64 px cube map
 * (cheap even in software GL, unlike a PMREM). In three r128 a plain cube environment only feeds
 * the specular term of MeshStandardMaterial — exactly what lets dark parts show their form
 * (Fresnel sheen, catch-lights) — while the lights below provide the diffuse shading.
 */
function studioEnvironment(renderer) {
  const scene = new Scene();
  const dome = new SphereGeometry(10, 48, 24);
  const pos = dome.attributes.position;
  const colors = [];
  const keyXY = new Vector3(KEY_DIR.x, KEY_DIR.y, 0).normalize();
  const dir = new Vector3();
  for (let i = 0; i < pos.count; i++) {
    dir.fromBufferAttribute(pos, i).normalize();
    const z = dir.z; // Z is up
    // Sky: bright overhead, dim at the horizon. Ground: dark, with a bounce on the key's side, so
    // the vertical faces of a dark part (which reflect the ground in an iso view) differ in tone.
    const facing = Math.max(0, dir.x * keyXY.x + dir.y * keyXY.y) / Math.max(Math.hypot(dir.x, dir.y), 1e-6);
    const v = z >= 0 ? 0.2 + 0.55 * z ** 0.8 : 0.05 + (0.1 + 0.12 * facing ** 1.5) * (1 + z) ** 2;
    colors.push(v * 0.96, v * 0.98, v);
  }
  dome.setAttribute('color', new Float32BufferAttribute(colors, 3));
  scene.add(new Mesh(dome, new MeshBasicMaterial({ vertexColors: true, side: BackSide, depthWrite: false })));
  const softbox = (dir, w, h, level) => {
    const m = new Mesh(new PlaneGeometry(w, h), new MeshBasicMaterial({ color: new Color(level, level, level * 0.97), side: DoubleSide }));
    m.position.copy(dir).normalize().multiplyScalar(8);
    m.up.set(0, 0, 1);
    m.lookAt(0, 0, 0);
    scene.add(m);
  };
  softbox(KEY_DIR.clone(), 7, 5, 0.7);
  softbox(new Vector3(-0.6, 1, 0.45), 9, 2.2, 0.75);
  const target = new WebGLCubeRenderTarget(64, { generateMipmaps: true, minFilter: LinearMipmapLinearFilter });
  const autoClear = renderer.autoClear;
  renderer.autoClear = true;
  new CubeCamera(0.1, 100, target).update(renderer, scene);
  renderer.autoClear = autoClear;
  scene.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
  return target.texture;
}

/** The eight corners of a Box3. */
export function boxCorners(box, out = []) {
  for (const x of [box.min.x, box.max.x]) {
    for (const y of [box.min.y, box.max.y]) {
      for (const z of [box.min.z, box.max.z]) out.push(new Vector3(x, y, z));
    }
  }
  return out;
}

export class Viewer {
  constructor(stage, canvas) {
    this.stage = stage;
    this.canvas = canvas;
    this.renderer = new WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.localClippingEnabled = true;
    this.renderer.autoClear = false; // panes and the axis gizmo clear explicitly
    this.renderer.setClearColor(BG);
    this.renderer.outputEncoding = sRGBEncoding;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFSoftShadowMap;
    this.renderer.shadowMap.autoUpdate = false; // one shadow pass per frame, shared by the quad panes

    this.scene = new Scene();
    this.scene.background = null;
    this.content = new Group(); // parts, ghosts and overlays live here
    this.scene.add(this.content);

    this.camera = new PerspectiveCamera(35, 1, 0.1, 1e5);
    this.camera.up.copy(Z_UP);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener('change', () => this.requestRender());

    // Orthographic cameras for the quad layout's top / front / right panes.
    this.orthos = ['top', 'front', 'right'].map(name => {
      const cam = new OrthographicCamera(-1, 1, 1, -1, 0.1, 1e5);
      cam.up.copy(name === 'top' ? new Vector3(0, 1, 0) : Z_UP);
      return { name, cam };
    });

    this._addLights();
    this.grid = new Group();
    this.scene.add(this.grid);
    this.floor = new Mesh(new PlaneGeometry(1, 1), new ShadowMaterial({ opacity: 0.36, depthWrite: false }));
    this.floor.receiveShadow = true;
    this.floor.renderOrder = -1;
    this.scene.add(this.floor);
    this._buildTriad();

    this.layout = 'single';
    this.bounds = new Box3(new Vector3(-50, -50, -50), new Vector3(50, 50, 50));
    this.L = 100;
    this.zoom = 1;
    this.labels = []; // {pos: Vector3, el: HTMLElement}
    this.labelLayer = stage.querySelector('#labels');
    this.quadLabels = stage.querySelector('#quad-labels');
    this._dirty = true;
    this._onFrame = null;

    new ResizeObserver(() => this.resize()).observe(stage);
    this.resize();
  }

  _addLights() {
    // Diffuse: hemisphere + three-point rig; specular: the studio cube (no PMREM, which costs
    // seconds in the software GL headless shots run on). The hemisphere's sky must be +Z:
    // HemisphereLight defaults to Y-up, which would light a part's −Y side with the ground colour.
    this.scene.environment = studioEnvironment(this.renderer);
    // Base-colour multipliers this gives an iso view: top ≈ 1.2, front (−Y) ≈ 0.7, side (+X) ≈ 0.45.
    const hemi = new HemisphereLight(srgb(0xdfe6f2), srgb(0x5a606a), 0.45);
    hemi.position.set(0, 0, 1);
    this.scene.add(hemi);
    this.scene.add(new AmbientLight(0xffffff, 0.04));
    this.key = new DirectionalLight(0xffffff, 0.6);
    this.sun = new DirectionalLight(0xffffff, 0.35); // the shadow caster
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0002;
    this.fill = new DirectionalLight(srgb(0xc8d6ff), 0.18); // back-left faces when orbiting
    this.rim = new DirectionalLight(srgb(0xfff1dc), 0.3);
    for (const l of [this.key, this.sun, this.fill, this.rim]) {
      this.scene.add(l, l.target);
    }
  }

  /** Small XYZ gizmo drawn in its own corner viewport. */
  _buildTriad() {
    this.triadScene = new Scene();
    this.triadCam = new OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
    this.triadCam.up.copy(Z_UP);
    const axes = [[1, 0, 0, 0xe5534b], [0, 1, 0, 0x57ab5a], [0, 0, 1, 0x539bf5]];
    for (const [x, y, z, c] of axes) {
      const arrow = new ArrowHelper(new Vector3(x, y, z), new Vector3(), 1.1, c, 0.32, 0.18);
      arrow.setColor(srgb(c));
      this.triadScene.add(arrow);
    }
  }

  /**
   * Size the grid to the model (its home-pose box, not the motion envelope, so a wide sweep does
   * not dwarf the parts) and put the floor under the home pose — a part that dips lower while
   * moving (a bucket digging) is drawn through the grid rather than the whole model hovering.
   * Light distances and the camera clip range come from the motion envelope.
   */
  setBounds(modelBox, envelope = modelBox) {
    this.bounds.copy(envelope);
    const size = modelBox.getSize(new Vector3());
    const center = modelBox.getCenter(new Vector3());
    const L = Math.max(envelope.getSize(new Vector3()).length(), 1);
    this.L = L;

    this.grid.clear();
    const extent = Math.max(size.x, size.y, size.z * 0.5, L * 0.05, 1);
    const step = niceStep(extent / 7);
    const half = Math.ceil((extent * 0.95 + step) / (step * 5)) * step * 5;
    const floorZ = modelBox.min.z - L * 0.002;
    const minor = new GridHelper(2 * half, Math.round((2 * half) / step));
    const major = new GridHelper(2 * half, Math.round((2 * half) / (step * 5)));
    minor.material.color = srgb(0x262b34);
    major.material.color = srgb(0x363d4a);
    const gx = Math.round(center.x / step) * step;
    const gy = Math.round(center.y / step) * step;
    for (const g of [minor, major]) {
      g.material.vertexColors = false;
      g.rotation.x = Math.PI / 2; // GridHelper lies in XZ; rotate into the XY plane
      g.position.set(gx, gy, floorZ);
      g.material.depthWrite = false;
      this.grid.add(g);
    }
    minor.renderOrder = -3;
    major.renderOrder = -2;
    this.floor.position.set(gx, gy, floorZ - L * 0.001);
    this.floor.scale.set(2 * half, 2 * half, 1);
    this.floorZ = floorZ;

    const place = (light, dir) => {
      light.position.copy(center).addScaledVector(dir.normalize(), L * 2);
      light.target.position.copy(center);
    };
    place(this.key, KEY_DIR.clone());
    place(this.fill, new Vector3(-1, 0.5, 0.2));
    place(this.rim, new Vector3(-0.4, 1, 0.25));
    this.setShadowBox(modelBox);

    this.camera.near = L / 500;
    this.camera.far = L * 100;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  /** Aim the shadow light's camera at `box` and the shadow it throws on the floor. */
  setShadowBox(box) {
    if (box.isEmpty()) return;
    const pts = boxCorners(box);
    const floorZ = this.floorZ ?? box.min.z;
    for (const p of [...pts]) {
      // Where this corner's shadow lands on the floor along the shadow light's direction.
      const t = (p.z - floorZ) / SHADOW_DIR.z;
      pts.push(p.clone().addScaledVector(SHADOW_DIR, -t));
    }
    const all = new Box3().setFromPoints(pts);
    const sphere = all.getBoundingSphere(new Sphere());
    const r = Math.max(sphere.radius * 1.05, 1e-3);
    this.sun.position.copy(sphere.center).addScaledVector(SHADOW_DIR, r * 3);
    this.sun.target.position.copy(sphere.center);
    const cam = this.sun.shadow.camera;
    Object.assign(cam, { left: -r, right: r, top: r, bottom: -r, near: r * 1.9, far: r * 4.1 });
    cam.updateProjectionMatrix();
    this.sun.shadow.normalBias = r * 0.0025;
    this.requestRender();
  }

  /** Caption of the main (top-left) pane in the quad layout. */
  setMainLabel(text) {
    this.quadLabels.firstElementChild.textContent = text.toUpperCase();
  }

  setLayout(layout) {
    this.layout = layout;
    this.quadLabels.hidden = layout !== 'quad';
    this.resize();
  }

  /**
   * Point the main camera along `dir` (target → camera) and fit `points` (world positions that
   * must be on screen) with a modest margin; `zoom` > 1 moves in (2 = twice as close). The ortho
   * panes are fitted to the same points.
   */
  frame(points, dir, zoom = 1) {
    if (!points.length) points = boxCorners(this.bounds);
    const d = (dir || this.camera.position.clone().sub(this.controls.target)).normalize();
    const box = new Box3().setFromPoints(points);
    const sphere = box.getBoundingSphere(new Sphere());
    const r = Math.max(sphere.radius, 1e-3);
    const f = d.clone().negate(); // viewing direction
    const u = f.clone().cross(this.camera.up);
    if (u.lengthSq() < 1e-12) u.copy(f).cross(new Vector3(0, 1, 0)); // looking straight along up
    u.normalize();
    const v = u.clone().cross(f);
    const tanV = Math.tan((this.camera.fov * Math.PI) / 360);
    const tanH = tanV * this.camera.aspect;

    const center = sphere.center.clone();
    const rel = new Vector3();
    let dist = r;
    for (let iter = 0; iter < 4; iter++) {
      dist = this.fitDistance(points, center, d, u, v, tanH, tanV);
      // Recentre so the projected extents sit symmetrically in the frame, then refit.
      const eye = center.clone().addScaledVector(d, dist);
      let x0 = Infinity; let x1 = -Infinity; let y0 = Infinity; let y1 = -Infinity;
      for (const p of points) {
        rel.copy(p).sub(eye);
        const depth = rel.dot(f);
        if (depth <= 1e-9) continue;
        const x = rel.dot(u) / (depth * tanH);
        const y = rel.dot(v) / (depth * tanV);
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
      if (!Number.isFinite(x0) || !Number.isFinite(y0)) break;
      const sx = ((x0 + x1) / 2) * tanH * dist;
      const sy = ((y0 + y1) / 2) * tanV * dist;
      center.addScaledVector(u, sx).addScaledVector(v, sy);
      if (Math.abs(sx) + Math.abs(sy) < r * 1e-3) break;
    }
    dist = Math.max(dist / Math.max(zoom, 1e-3), r * 0.02);
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(d, dist);
    this.camera.near = Math.max(dist / 1000, r / 500);
    this.camera.far = dist + r * 50 + this.L * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.setFitPoints(points, zoom);
    this.requestRender();
  }

  /**
   * Distance from `center` along unit `d` (target → camera) at which every point projects inside
   * the perspective view at FILL_PERSPECTIVE of the pane; `u`/`v` are the screen axes.
   */
  fitDistance(points, center, d, u, v, tanH, tanV) {
    const p = new Vector3();
    let dist = 0;
    for (const q of points) {
      p.copy(q).sub(center);
      const depth = p.dot(d); // how far the point sits toward the camera
      dist = Math.max(dist, depth + Math.abs(p.dot(u)) / (tanH * FILL_PERSPECTIVE),
        depth + Math.abs(p.dot(v)) / (tanV * FILL_PERSPECTIVE));
    }
    return dist;
  }

  /** Remember what the ortho panes should frame (quad layout) and fit them. */
  setFitPoints(points, zoom = 1) {
    this._fitPoints = points.map(p => p.clone());
    this.zoom = zoom;
    this.fitOrthos(this._fitPoints, zoom);
  }

  /**
   * Fit the quad layout's orthographic panes to `points`. Each camera sits outside the whole
   * motion envelope so a focused pair is never cut by the near plane or the parts around it
   * clipped away.
   */
  fitOrthos(points, zoom = 1) {
    if (!points || !points.length) points = boxCorners(this.bounds);
    const box = new Box3().setFromPoints(points);
    const center = box.getCenter(new Vector3());
    const envelope = this.bounds.getBoundingSphere(new Sphere());
    const dist = center.distanceTo(envelope.center) + envelope.radius + box.getSize(new Vector3()).length() + 10;
    const q = new Vector3();
    for (const { name, cam } of this.orthos) {
      const dir = VIEW_DIRS[name];
      cam.position.copy(center).addScaledVector(dir, dist);
      cam.lookAt(center);
      cam.updateMatrixWorld();
      // Extent of the points in the camera's screen axes.
      const inv = cam.matrixWorldInverse;
      let x0 = Infinity; let x1 = -Infinity; let y0 = Infinity; let y1 = -Infinity;
      for (const p of points) {
        q.copy(p).applyMatrix4(inv);
        x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y);
      }
      const aspect = this._quadAspect || 1;
      const halfH = (Math.max((y1 - y0) / 2, (x1 - x0) / 2 / aspect) / FILL_ORTHO) / Math.max(zoom, 1e-3) + 1e-3;
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      Object.assign(cam, { left: cx - halfH * aspect, right: cx + halfH * aspect, top: cy + halfH, bottom: cy - halfH });
      cam.near = 0.01;
      cam.far = dist + envelope.radius * 4 + 10;
      cam.updateProjectionMatrix();
    }
  }

  resize() {
    const w = Math.max(this.stage.clientWidth, 1);
    const h = Math.max(this.stage.clientHeight, 1);
    // Supersample ×2 whatever the display (edge lines stay smooth in 1× screenshots, the canvas
    // is scaled down by CSS), capped at ~8 Mpx drawn so a large window stays interactive.
    this.renderer.setPixelRatio(Math.max(1, Math.min(2, Math.sqrt(MAX_DRAWN_PX / (w * h)))));
    this.renderer.setSize(w, h, false);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.width = w;
    this.height = h;
    // Quad panes are half width and half height, so every pane keeps the canvas aspect.
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._quadAspect = w / h;
    this.fitOrthos(this._fitPoints, this.zoom);
    this.requestRender();
  }

  /** Screen rects {x, y, w, h} (CSS px, origin top-left) and cameras of the visible panes. */
  panes() {
    const { width: w, height: h } = this;
    if (this.layout !== 'quad') return [{ name: 'iso', cam: this.camera, x: 0, y: 0, w, h }];
    const hw = Math.floor(w / 2);
    const hh = Math.floor(h / 2);
    const rects = [[0, 0], [hw, 0], [0, hh], [hw, hh]];
    const cams = [{ name: 'iso', cam: this.camera }, ...this.orthos];
    return cams.map((c, i) => ({ ...c, x: rects[i][0], y: rects[i][1], w: i % 2 ? w - hw : hw, h: i < 2 ? hh : h - hh }));
  }

  addLabel(el) {
    const label = { pos: new Vector3(), el };
    this.labelLayer.appendChild(el);
    this.labels.push(label);
    return label;
  }

  removeLabel(label) {
    label.el.remove();
    this.labels = this.labels.filter(l => l !== label);
  }

  requestRender() {
    this._dirty = true;
  }

  /** Start the render loop; `onFrame(dt)` runs every animation frame (playback). */
  start(onFrame) {
    this._onFrame = onFrame;
    let last = performance.now();
    const loop = now => {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      if (this._onFrame) this._onFrame(dt);
      if (this._dirty) this.render();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  render() {
    this._dirty = false;
    const r = this.renderer;
    const panes = this.panes();
    const H = this.height;
    r.shadowMap.needsUpdate = true; // rendered by the first pane, reused by the others
    r.setScissorTest(true);
    if (this.layout === 'quad') {
      r.setViewport(0, 0, this.width, H);
      r.setScissor(0, 0, this.width, H);
      r.setClearColor(DIVIDER);
      r.clear();
    }
    r.setClearColor(BG);
    for (const p of panes) {
      const inset = this.layout === 'quad' ? 1 : 0;
      const x = p.x + inset;
      const y = H - (p.y + p.h) + inset; // WebGL viewports are bottom-left based
      const w = p.w - 2 * inset;
      const h = p.h - 2 * inset;
      if (p.cam.isPerspectiveCamera && Math.abs(p.cam.aspect - w / h) > 1e-6) {
        p.cam.aspect = w / h;
        p.cam.updateProjectionMatrix();
      }
      r.setViewport(x, y, w, h);
      r.setScissor(x, y, w, h);
      r.clear();
      // The ground shadow grounds the perspective view; seen straight down in the top pane it
      // is just a dark blot over the parts' outline (and edge-on in front/right).
      this.floor.visible = p.cam === this.camera;
      r.render(this.scene, p.cam);
      this._renderTriad(p.cam, x, y);
    }
    r.setScissorTest(false);
    this._placeLabels(panes[0]);
  }

  _renderTriad(cam, x, y) {
    const s = 76;
    const dir = new Vector3(0, 0, 1).applyQuaternion(cam.quaternion);
    this.triadCam.position.copy(dir.multiplyScalar(4));
    this.triadCam.quaternion.copy(cam.quaternion);
    this.triadCam.updateMatrixWorld();
    this.renderer.setViewport(x + 6, y + 6, s, s);
    this.renderer.setScissor(x + 6, y + 6, s, s);
    this.renderer.clearDepth();
    this.renderer.render(this.triadScene, this.triadCam);
  }

  /** Project world-anchored DOM labels through the primary (iso) pane. */
  _placeLabels(pane) {
    const cam = pane.cam;
    const v = new Vector3();
    for (const l of this.labels) {
      v.copy(l.pos).project(cam);
      const onScreen = v.z < 1 && Math.abs(v.x) <= 1.05 && Math.abs(v.y) <= 1.05;
      l.el.style.display = onScreen ? '' : 'none';
      if (!onScreen) continue;
      const px = pane.x + ((v.x + 1) / 2) * pane.w;
      const py = pane.y + ((1 - v.y) / 2) * pane.h;
      l.el.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    }
  }
}
