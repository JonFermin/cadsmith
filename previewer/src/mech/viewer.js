// Renderer, cameras, lights, grid and viewport layout (single or 2×2 quad). Z is up.
import {
  AmbientLight, ArrowHelper, Box3, DirectionalLight, GridHelper, Group, HemisphereLight,
  OrthographicCamera, PerspectiveCamera, Scene, Sphere, Vector3, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const BG = 0x15171c;
const DIVIDER = 0x08090b;
const Z_UP = new Vector3(0, 0, 1);

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

export class Viewer {
  constructor(stage, canvas) {
    this.stage = stage;
    this.canvas = canvas;
    this.renderer = new WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.localClippingEnabled = true;
    this.renderer.autoClear = false; // panes and the axis gizmo clear explicitly
    this.renderer.setClearColor(BG);

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
    this._buildTriad();

    this.layout = 'single';
    this.bounds = new Box3(new Vector3(-50, -50, -50), new Vector3(50, 50, 50));
    this.labels = []; // {pos: Vector3, el: HTMLElement}
    this.labelLayer = stage.querySelector('#labels');
    this.quadLabels = stage.querySelector('#quad-labels');
    this._dirty = true;
    this._onFrame = null;

    new ResizeObserver(() => this.resize()).observe(stage);
    this.resize();
  }

  _addLights() {
    this.scene.add(new HemisphereLight(0xe6ecff, 0x2b2f38, 0.65));
    this.scene.add(new AmbientLight(0xffffff, 0.08));
    this.key = new DirectionalLight(0xffffff, 0.75);
    this.fill = new DirectionalLight(0xc8d6ff, 0.28);
    this.rim = new DirectionalLight(0xffffff, 0.18);
    for (const l of [this.key, this.fill, this.rim]) {
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
      this.triadScene.add(new ArrowHelper(new Vector3(x, y, z), new Vector3(), 1.1, c, 0.32, 0.18));
    }
  }

  /** Size the grid, lights and clip planes to the model's bounds (home + motion envelope). */
  setBounds(box) {
    this.bounds.copy(box);
    const size = box.getSize(new Vector3());
    const center = box.getCenter(new Vector3());
    const L = Math.max(size.length(), 1);
    this.L = L;

    this.grid.clear();
    const step = niceStep(L / 12);
    const half = Math.ceil((Math.max(size.x, size.y) * 0.5 + L * 0.6) / (step * 5)) * step * 5;
    const minor = new GridHelper(2 * half, Math.round((2 * half) / step), 0x2a2f39, 0x22262e);
    const major = new GridHelper(2 * half, Math.round((2 * half) / (step * 5)), 0x3a4150, 0x333946);
    for (const g of [minor, major]) {
      g.rotation.x = Math.PI / 2; // GridHelper lies in XZ; rotate into the XY plane
      g.position.set(Math.round(center.x / step) * step, Math.round(center.y / step) * step, box.min.z - L * 0.002);
      g.material.depthWrite = false;
      this.grid.add(g);
    }
    minor.renderOrder = -2;
    major.renderOrder = -1;

    const place = (light, dir) => {
      light.position.copy(center).addScaledVector(dir.normalize(), L * 2);
      light.target.position.copy(center);
    };
    place(this.key, new Vector3(0.55, -0.8, 1.3));
    place(this.fill, new Vector3(-1, 0.7, 0.35));
    place(this.rim, new Vector3(-0.2, 1, -0.6));

    this.camera.near = L / 500;
    this.camera.far = L * 100;
    this.camera.updateProjectionMatrix();
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

  /** Point the main camera along `dir` (target → camera) and fit `box`. */
  frame(box, dir) {
    const d = (dir || this.camera.position.clone().sub(this.controls.target)).normalize();
    const sphere = box.getBoundingSphere(new Sphere());
    const r = Math.max(sphere.radius, 1e-3);
    const dist = Math.max(this.fitDistance(box, sphere.center, d), r * 0.5);
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).addScaledVector(d, dist);
    this.camera.near = Math.max(dist / 1000, r / 200);
    this.camera.far = dist + r * 50 + this.L * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.fitOrthos(box);
    this.requestRender();
  }

  /**
   * Distance from `center` along unit `d` (target → camera) at which all eight corners of `box`
   * fit the perspective view with a margin. Tighter than a bounding-sphere fit for flat or
   * elongated mechanisms (linkages), which a sphere frames far too loosely.
   */
  fitDistance(box, center, d) {
    const MARGIN = 1.12;
    const tanV = Math.tan((this.camera.fov * Math.PI) / 360);
    const tanH = tanV * this.camera.aspect;
    const f = d.clone().negate(); // viewing direction
    const u = f.clone().cross(this.camera.up);
    if (u.lengthSq() < 1e-12) u.copy(f).cross(new Vector3(0, 1, 0)); // looking straight along up
    u.normalize();
    const v = u.clone().cross(f);
    const p = new Vector3();
    let dist = 0;
    for (const x of [box.min.x, box.max.x]) {
      for (const y of [box.min.y, box.max.y]) {
        for (const z of [box.min.z, box.max.z]) {
          p.set(x, y, z).sub(center);
          const depth = p.dot(d); // how far the corner sits toward the camera
          dist = Math.max(dist, depth + (MARGIN * Math.abs(p.dot(u))) / tanH,
            depth + (MARGIN * Math.abs(p.dot(v))) / tanV);
        }
      }
    }
    return dist;
  }

  /** Fit the quad layout's orthographic panes to `box`. */
  fitOrthos(box) {
    const center = box.getCenter(new Vector3());
    const size = box.getSize(new Vector3());
    const dist = size.length() * 2 + 10;
    for (const { name, cam } of this.orthos) {
      const dir = VIEW_DIRS[name];
      cam.position.copy(center).addScaledVector(dir, dist);
      cam.lookAt(center);
      cam.updateMatrixWorld();
      // Extent of the box in the camera's screen axes.
      const inv = cam.matrixWorldInverse;
      const pb = box.clone().applyMatrix4(inv);
      const w = pb.max.x - pb.min.x;
      const h = pb.max.y - pb.min.y;
      const aspect = this._quadAspect || 1;
      const halfH = Math.max(h / 2, w / 2 / aspect) * 1.18 + 1e-3;
      const cx = (pb.max.x + pb.min.x) / 2;
      const cy = (pb.max.y + pb.min.y) / 2;
      Object.assign(cam, { left: cx - halfH * aspect, right: cx + halfH * aspect, top: cy + halfH, bottom: cy - halfH });
      cam.near = 0.01;
      cam.far = dist * 2 + size.length() * 2;
      cam.updateProjectionMatrix();
    }
  }

  resize() {
    const w = Math.max(this.stage.clientWidth, 1);
    const h = Math.max(this.stage.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.width = w;
    this.height = h;
    // Quad panes are half width and half height, so every pane keeps the canvas aspect.
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._quadAspect = w / h;
    this.fitOrthos(this._orthoBox || this.bounds);
    this.requestRender();
  }

  /** Remember what the ortho panes should frame (quad layout). */
  setOrthoBox(box) {
    this._orthoBox = box.clone();
    this.fitOrthos(box);
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
